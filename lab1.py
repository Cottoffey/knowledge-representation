"""
Лабораторная работа 1. Извлечение и структурирование знаний из текста.
Пайплайн: предобработка -> NER -> извлечение отношений -> граф знаний ->
запросы (networkx + SPARQL) -> визуализация -> оценка качества.

Установка:  pip install natasha networkx rdflib matplotlib
Запуск:     python lab1_knowledge_graph.py [файл_с_текстом.txt]
"""
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
from natasha import (Doc, Segmenter, MorphVocab, NewsEmbedding,
                     NewsMorphTagger, NewsSyntaxParser, NewsNERTagger)
from rdflib import Graph, Literal, Namespace, RDF, RDFS

# ----------------------------------------------------------------------
# 0. Входные данные
# ----------------------------------------------------------------------
SAMPLE_TEXT = """
Сергей Брин и Ларри Пейдж основали компанию Google в 1998 году в Калифорнии.
Компания Google находится в Маунтин-Вью. Сундар Пичаи возглавляет Google с 2015 года.
Google купил YouTube в 2006 году. Google приобрел также компанию Android.
Илон Маск основал компанию SpaceX в Калифорнии. Илон Маск возглавляет Tesla.
Tesla расположена в Остине. Стив Джобс основал компанию Apple в Купертино.
Тим Кук возглавляет Apple. Apple выпустила iPhone в 2007 году.
"""

# Эталонная разметка для оценки (составляется вручную по тексту!)
GOLD_ENTITIES = {
    ("Сергей Брин", "PER"), ("Ларри Пейдж", "PER"), ("Google", "ORG"),
    ("Калифорния", "LOC"), ("Маунтин-Вью", "LOC"), ("Сундар Пичаи", "PER"),
    ("YouTube", "ORG"), ("Android", "ORG"), ("Илон Маск", "PER"),
    ("SpaceX", "ORG"), ("Tesla", "ORG"), ("Остин", "LOC"),
    ("Стив Джобс", "PER"), ("Apple", "ORG"), ("Купертино", "LOC"),
    ("Тим Кук", "PER"),
}
GOLD_TRIPLES = {
    ("Сергей Брин", "основать", "Google"),
    ("Ларри Пейдж", "основать", "Google"),
    ("Google", "находиться в", "Маунтин-Вью"),
    ("Сундар Пичаи", "возглавлять", "Google"),
    ("Google", "купить", "YouTube"),
    ("Google", "приобрести", "Android"),
    ("Илон Маск", "основать", "SpaceX"),
    ("Илон Маск", "возглавлять", "Tesla"),
    ("Tesla", "расположить в", "Остин"),
    ("Стив Джобс", "основать", "Apple"),
    ("Тим Кук", "возглавлять", "Apple"),
}

# ----------------------------------------------------------------------
# 1. Предобработка
# ----------------------------------------------------------------------
segmenter = Segmenter()
morph_vocab = MorphVocab()
emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)


def preprocess(text: str) -> str:
    """Очистка: пробелы, переносы строк, «ёлочки» -> обычные кавычки."""
    text = text.replace("«", '"').replace("»", '"')
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def analyze(text: str) -> Doc:
    """Сегментация, морфология (лемматизация), синтаксис, NER."""
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for tok in doc.tokens:
        tok.lemmatize(morph_vocab)
    doc.parse_syntax(syntax_parser)
    doc.tag_ner(ner_tagger)
    for span in doc.spans:
        span.normalize(morph_vocab)       # «Калифорнии» -> «Калифорния»
    return doc


# ----------------------------------------------------------------------
# 2. Извлечение сущностей (NER)
# ----------------------------------------------------------------------
def get_entities(doc: Doc):
    return [(s.normal, s.type, s.start, s.stop) for s in doc.spans]


# ----------------------------------------------------------------------
# 3. Извлечение отношений (правила по дереву зависимостей)
# ----------------------------------------------------------------------
SUBJ_RELS = {"nsubj", "nsubj:pass"}
OBJ_RELS = {"obj", "iobj", "obl", "nmod"}


def expand_conj(tok_id, children):
    """Токен + все токены, связанные с ним через conj («Брин и Пейдж»)."""
    result, stack = [tok_id], [tok_id]
    while stack:
        cur = stack.pop()
        for ch in children[cur]:
            if ch["rel"] == "conj":
                result.append(ch["id"])
                stack.append(ch["id"])
    return result


def extract_relations(doc: Doc):
    triples = []
    last_per = None          # простейшая кореференция: последняя названная персона
    for sent in doc.sents:
        toks = {t.id: t for t in sent.tokens}
        children = defaultdict(list)
        for t in sent.tokens:
            children[t.head_id].append({"id": t.id, "rel": t.rel, "tok": t})

        # head-токен каждой сущности -> сущность
        ent_by_head = {}
        for span in sent.spans:
            ids = {t.id for t in sent.tokens
                   if t.start >= span.start and t.stop <= span.stop}
            for tid in ids:
                if toks[tid].head_id not in ids:
                    ent_by_head[tid] = (span.normal, span.type)

        def resolve(tok_id):
            """Токен -> сущности. Если токен — нарицательное слово
            («компанию»), ищем сущность среди зависимых (appos/nmod/flat)."""
            if tok_id in ent_by_head:
                return [ent_by_head[tok_id]]
            found = []
            for ch in children[tok_id]:
                if ch["rel"] in {"appos", "nmod", "flat", "flat:name"}:
                    found += resolve(ch["id"])
            return found

        for verb in sent.tokens:
            if verb.pos not in {"VERB", "ADJ"} or not children[verb.id]:
                continue
            subjects, objects = [], []
            for ch in children[verb.id]:
                if ch["rel"] in SUBJ_RELS:
                    if ch["tok"].pos == "PRON" and ch["tok"].lemma in {"он", "она"} and last_per:
                        subjects.append(last_per)    # «он» -> последняя персона
                    for i in expand_conj(ch["id"], children):
                        subjects += resolve(i)
                elif ch["rel"] in OBJ_RELS:
                    # предлог («в», «с»), если есть, входит в название отношения
                    prep = next((toks[c["id"]].text.lower()
                                 for c in children[ch["id"]] if c["rel"] == "case"), "")
                    for i in expand_conj(ch["id"], children):
                        objects += [(e, prep) for e in resolve(i)]
            if not subjects and objects and last_per:
                subjects.append(last_per)            # пропущенное подлежащее
            for s in subjects:
                for o, prep in objects:
                    rel = f"{verb.lemma} {prep}".strip()
                    triples.append((s, rel, o, sent.text))
        pers = [(sp.normal, sp.type) for sp in sent.spans if sp.type == "PER"]
        if pers:
            last_per = pers[0]
    return triples


# ----------------------------------------------------------------------
# 4. Граф знаний (networkx + RDF)
# ----------------------------------------------------------------------
def build_graph(triples):
    G = nx.MultiDiGraph()
    for (s, st), rel, (o, ot), sent in triples:
        G.add_node(s, type=st)
        G.add_node(o, type=ot)
        if not G.has_edge(s, o, key=rel):
            G.add_edge(s, o, key=rel, label=rel, sentence=sent)
    return G


def build_rdf(G):
    EX = Namespace("http://example.org/kg/")
    rdf = Graph()
    rdf.bind("ex", EX)
    for cls in ("PER", "ORG", "LOC"):          # простая онтология классов
        rdf.add((EX[cls], RDF.type, RDFS.Class))
    for n, d in G.nodes(data=True):
        uri = EX[re.sub(r"\W+", "_", n)]
        rdf.add((uri, RDF.type, EX[d["type"]]))
        rdf.add((uri, RDFS.label, Literal(n, lang="ru")))
    for s, o, rel in G.edges(keys=True):
        p = EX[re.sub(r"\W+", "_", rel)]
        rdf.add((EX[re.sub(r"\W+", "_", s)], p, EX[re.sub(r"\W+", "_", o)]))
    return rdf, EX


# ----------------------------------------------------------------------
# 5. Запросы и визуализация
# ----------------------------------------------------------------------
def run_queries(G, rdf, EX):
    print("\n=== Запрос 1: все факты про Google ===")
    for s, o, rel in G.edges("Google", keys=True):
        print(f"  Google --{rel}--> {o}")
    for s, o, rel in G.in_edges("Google", keys=True):
        print(f"  {s} --{rel}--> Google")

    print("\n=== Запрос 2: кто основал какие компании ===")
    for s, o, rel in G.edges(keys=True):
        if rel == "основать":
            print(f"  {s} -> {o}")

    print("\n=== Запрос 3: степень центральности узлов (топ-5) ===")
    cent = nx.degree_centrality(G)
    for n, c in sorted(cent.items(), key=lambda x: -x[1])[:5]:
        print(f"  {n}: {c:.2f}")

    print("\n=== Запрос 4: путь Сергей Брин -> Маунтин-Вью ===")
    try:
        print("  ", nx.shortest_path(G.to_undirected(), "Сергей Брин", "Маунтин-Вью"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        print("   пути нет")

    print("\n=== Запрос 5 (SPARQL): все люди и компании, которые они возглавляют ===")
    q = """
    PREFIX ex: <http://example.org/kg/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    SELECT ?pl ?ol WHERE {
        ?p a ex:PER ; ex:возглавлять ?o ; rdfs:label ?pl .
        ?o rdfs:label ?ol .
    }"""
    for row in rdf.query(q):
        print(f"  {row.pl} возглавляет {row.ol}")


def visualize(G, path="knowledge_graph.png"):
    colors = {"PER": "#6baed6", "ORG": "#fd8d3c", "LOC": "#74c476"}
    pos = nx.spring_layout(G, seed=42, k=1.2)
    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=1800,
                           node_color=[colors.get(G.nodes[n]["type"], "gray") for n in G])
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, edge_color="gray")
    edge_labels = {(u, v): k for u, v, k in G.edges(keys=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=t)
               for t, c in colors.items()]
    plt.legend(handles=handles)
    plt.title("Граф знаний")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nГраф сохранён: {path}")


# ----------------------------------------------------------------------
# 6. Оценка качества
# ----------------------------------------------------------------------
def prf(pred: set, gold: set):
    tp = len(pred & gold)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(entities, triples):
    pred_e = {(n, t) for n, t, *_ in entities}
    pred_t = {(s[0], rel, o[0]) for s, rel, o, _ in triples}
    gold_e = {(n.lower(), t) for n, t in GOLD_ENTITIES}
    gold_t = {(s.lower(), r, o.lower()) for s, r, o in GOLD_TRIPLES}
    pred_e = {(n.lower(), t) for n, t in pred_e}
    pred_t = {(s.lower(), r, o.lower()) for s, r, o in pred_t}

    for name, pred, gold in (("NER (сущность+тип)", pred_e, gold_e),
                             ("Отношения (триплеты)", pred_t, gold_t)):
        p, r, f = prf(pred, gold)
        print(f"\n{name}: P={p:.2f}  R={r:.2f}  F1={f:.2f}")
        print("  Ложные срабатывания (FP):", sorted(pred - gold))
        print("  Пропущено (FN):          ", sorted(gold - pred))


# ----------------------------------------------------------------------
if __name__ == "__main__":
    raw = open(sys.argv[1], encoding="utf-8").read() if len(sys.argv) > 1 else SAMPLE_TEXT
    text = preprocess(raw)
    doc = analyze(text)

    entities = get_entities(doc)
    print("=== Сущности ===")
    for n, t, *_ in entities:
        print(f"  {n}\t[{t}]")

    triples = extract_relations(doc)
    print("\n=== Триплеты (субъект, отношение, объект) ===")
    for s, rel, o, _ in triples:
        print(f"  ({s[0]}, {rel}, {o[0]})")

    G = build_graph(triples)
    rdf, EX = build_rdf(G)
    rdf.serialize("knowledge_graph.ttl", format="turtle")
    print(f"\nУзлов: {G.number_of_nodes()}, рёбер: {G.number_of_edges()}")

    run_queries(G, rdf, EX)
    visualize(G)
    evaluate(entities, triples)
