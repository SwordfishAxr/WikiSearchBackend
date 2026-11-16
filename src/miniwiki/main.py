#!/usr/bin/env python3
"""
Гибридный поиск по дампу Википедии: SQLite FTS5 (BM25) + qdrant (bge-m3) + RRF.

  python wiki_search.py index simplewiki-latest-pages-articles.xml.bz2 --lang english
  python wiki_search.py embed --intro-only
  python wiki_search.py search "домашнее животное"
"""
import argparse
import bz2
import os
import re
import sqlite3
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import mwparserfromhell
import requests
import snowballstemmer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, PointStruct, ScalarQuantization, ScalarQuantizationConfig,
    ScalarType, VectorParams,
)
from tqdm import tqdm

from http.server import HTTPServer, BaseHTTPRequestHandler;
import json;

DB_PATH = "wiki.db"
COLLECTION = "wiki"
# по одному llama-server (-t 1 -np 1) на порт; запросы раздаются по кругу
LLAMA_PORTS = [8000, 8001, 8002, 8003]
LLAMA_URLS = [f"http://localhost:{p}/v1/embeddings" for p in LLAMA_PORTS]
QDRANT_URL = "http://localhost:6333"

# префиксы: для e5 нужны "passage: " / "query: ", для bge-m3 поставь ""
PASSAGE_PREFIX = ""
QUERY_PREFIX = ""

MAX_WORDS = 150      # слов в чанке (для -c 4096 -np 4: на слот 1024 токена)
WINDOW = 2048        # сколько чанков сортируем по длине за раз
BATCH = 16           # текстов в одном запросе к llama-server
WORKERS = len(LLAMA_PORTS)   # по одному запросу на сервер


# ---------- общее ----------

def connect():
    db = sqlite3.connect(DB_PATH)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS chunks(
            id INTEGER PRIMARY KEY, title TEXT, pos INTEGER, text TEXT);
        CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
        CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(stem);
    """)
    return db


def meta_get(db, k, default=None):
    r = db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    return r[0] if r else default


def meta_set(db, k, v):
    db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, str(v)))
    db.commit()


def tokenize(s):
    return re.findall(r"\w+", s.lower())


MAX_CHARS = 1500     # жёсткий потолок длины текста, защита от мусорных чанков


def _parse(r):
    data = sorted(r.json()["data"], key=lambda x: x["index"])
    return [d["embedding"] for d in data]


def embed(texts, prefix="", url=None):
    url = url or LLAMA_URLS[0]
    texts = [prefix + t[:MAX_CHARS] for t in texts]
    r = requests.post(url, json={"input": texts}, timeout=600)
    if r.ok:
        return _parse(r)
    # сервер отверг батч (обычно слишком длинный вход): по одному,
    # укорачивая текст вдвое, пока не примет
    out = []
    for t in texts:
        while True:
            r = requests.post(url, json={"input": [t]}, timeout=600)
            if r.ok:
                out.append(_parse(r)[0])
                break
            if len(t) < 50:
                raise RuntimeError(r.text)
            t = t[: len(t) // 2]
    return out


def rrf(*rankings, k=60):
    score = {}
    for ranking in rankings:
        for pos, doc_id in enumerate(ranking):
            score[doc_id] = score.get(doc_id, 0) + 1 / (k + pos + 1)
    return sorted(score, key=score.get, reverse=True)


# ---------- index: парсинг, нарезка, FTS5 ----------

def pages(path):
    """Потоково читает дамп, отдаёт (title, wikitext) для обычных статей."""
    with bz2.open(path, "rb") as f:
        root = None
        title = ns = text = None
        redirect = False
        for ev, el in ET.iterparse(f, events=("start", "end")):
            if root is None:
                root = el
            if ev != "end":
                continue
            tag = el.tag.rsplit("}", 1)[-1]
            if tag == "title":
                title = el.text
            elif tag == "ns":
                ns = el.text
            elif tag == "redirect":
                redirect = True
            elif tag == "text":
                text = el.text
            elif tag == "page":
                if ns == "0" and not redirect and text:
                    yield title, text
                title = ns = text = None
                redirect = False
                root.clear()


_stemmer = None


def work(args):
    """Выполняется в пулe процессов: чистит разметку, режет, стеммит."""
    global _stemmer
    (title, text), lang, max_words = args
    if _stemmer is None:
        _stemmer = snowballstemmer.stemmer(lang)
    try:
        plain = mwparserfromhell.parse(text).strip_code()
    except Exception:
        return []
    paras = [p.strip() for p in plain.split("\n")]
    paras = [p for p in paras if len(p.split()) >= 5]
    # режем слишком длинные абзацы по словам
    pieces = []
    for p in paras:
        w = p.split()
        for i in range(0, len(w), max_words):
            pieces.append(" ".join(w[i:i + max_words]))
    paras = pieces

    chunks, cur, n = [], [], 0
    for p in paras:
        w = len(p.split())
        if cur and n + w > max_words:
            chunks.append("\n".join(cur))
            cur, n = [], 0
        cur.append(p)
        n += w
    if cur:
        chunks.append("\n".join(cur))

    out = []
    for pos, c in enumerate(chunks):
        full = f"{title}\n{c}"
        stem = " ".join(_stemmer.stemWords(tokenize(full)))
        out.append((title, pos, full, stem))
    return out


def cmd_index(a):
    db = connect()
    if db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]:
        sys.exit(f"{DB_PATH} уже содержит данные. Удали файл, чтобы переиндексировать.")
    meta_set(db, "lang", a.lang)

    jobs = ((p, a.lang, MAX_WORDS) for p in pages(a.dump))
    n = 0
    with Pool(os.cpu_count() or 2) as pool:
        for res in tqdm(pool.imap(work, jobs, chunksize=8), unit=" статей"):
            for title, pos, text, stem in res:
                cur = db.execute(
                    "INSERT INTO chunks(title,pos,text) VALUES(?,?,?)",
                    (title, pos, text))
                db.execute("INSERT INTO fts(rowid,stem) VALUES(?,?)",
                           (cur.lastrowid, stem))
            n += 1
            if n % 2000 == 0:
                db.commit()
    db.commit()
    total = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    print(f"Готово: {total} чанков")


# ---------- embed: эмбеддинги в qdrant с чекпоинтами ----------

def cmd_embed(a):
    db = connect()
    client = QdrantClient(url=QDRANT_URL, timeout=120)

    if not client.collection_exists(COLLECTION):
        dim = len(embed(["тест"])[0])
        print(f"Размер вектора: {dim}")
        client.create_collection(
            COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE,
                                        on_disk=True),
            quantization_config=ScalarQuantization(
                scalar=ScalarQuantizationConfig(type=ScalarType.INT8,
                                                always_ram=True)),
        )

    key = "emb_last_intro" if a.intro_only else "emb_last_all"
    last = int(meta_get(db, key, 0))
    cond = "id>?" + (" AND pos=0" if a.intro_only else "")
    total = db.execute(f"SELECT COUNT(*) FROM chunks WHERE {cond}",
                       (last,)).fetchone()[0]
    print(f"К обработке: {total} чанков (с id>{last})")

    def job(arg):
        i, batch = arg
        vecs = embed([r[2] for r in batch], PASSAGE_PREFIX,
                     LLAMA_URLS[i % len(LLAMA_URLS)])
        client.upsert(COLLECTION, points=[
            PointStruct(id=r[0], vector=v, payload={"title": r[1]})
            for r, v in zip(batch, vecs)])
        return len(batch)

    bar = tqdm(total=total, unit=" чанков")
    while True:
        rows = db.execute(
            f"SELECT id,title,text FROM chunks WHERE {cond} ORDER BY id LIMIT ?",
            (last, WINDOW)).fetchall()
        if not rows:
            break
        by_len = sorted(rows, key=lambda r: len(r[2]))
        batches = [by_len[i:i + BATCH] for i in range(0, len(by_len), BATCH)]
        with ThreadPoolExecutor(WORKERS) as ex:
            for n in ex.map(job, enumerate(batches)):
                bar.update(n)
        last = rows[-1][0]
        meta_set(db, key, last)   # чекпоинт после каждого окна
    bar.close()
    print("Готово")


# ---------- search: BM25 + векторы + RRF ----------

def search(query, n):
    db = connect()
    lang = meta_get(db, "lang", "english")
    stems = snowballstemmer.stemmer(lang).stemWords(tokenize(query))

    bm_ids = []
    if stems:
        q = " OR ".join(f'"{t}"' for t in stems)
        bm_ids = [r[0] for r in db.execute(
            "SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT 50",
            (q,))]

    client = QdrantClient(url=QDRANT_URL, timeout=60)
    hits = client.query_points(COLLECTION, query=embed([query], QUERY_PREFIX)[0],
                               limit=50).points
    vec_ids = [h.id for h in hits]

    for i in rrf(bm_ids, vec_ids)[:n]:
        title, text = db.execute(
            "SELECT title, text FROM chunks WHERE id=?", (i,)).fetchone()
        src = ("B" if i in bm_ids else "-") + ("V" if i in vec_ids else "-")
        snippet = text.split("\n", 1)[-1][:200].replace("\n", " ")
        yield { "src": src, "title": title, "snippet": snippet };

def cmd_search(a):
    for i in search(" ".join(a.query), a.n):
        src, title, snippet = i["src"], i["title"], i["snippet"];
        print(f"[{src}] {title}: {snippet}...\n");

class SearchApi(BaseHTTPRequestHandler):
    def do_POST(self):
        cl = int(self.headers["Content-Length"]);
        req = json.loads(self.rfile.read(cl));
        results = list(search(req["query"], req.get("n", 10)));
        res = json.dumps({ "result": results }, separators=",:").encode();

        self.send_response(200);
        self.send_header("Content-Length", str(len(res)));
        self.end_headers();
        self.wfile.write(res);

def cmd_serve(a):
    addr = ("0.0.0.0", 8801);
    httpd = HTTPServer(addr, SearchApi);
    print(f"Starting server on port {addr[0]}:{addr[1]}... Press Ctrl+C to stop.");
    try:
        httpd.serve_forever();
    except KeyboardInterrupt:
        print("\nStopping server...");
        httpd.server_close();

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("index")
    s.add_argument("dump")
    s.add_argument("--lang", default="russian",
                   help="russian | english (язык стеммера)")
    s.set_defaults(f=cmd_index)

    s = sub.add_parser("embed")
    s.add_argument("--intro-only", action="store_true",
                   help="только первый чанк каждой статьи")
    s.set_defaults(f=cmd_embed)

    s = sub.add_parser("search")
    s.add_argument("query", nargs="+")
    s.add_argument("-n", type=int, default=10)
    s.set_defaults(f=cmd_search)

    s = sub.add_parser("serve");
    s.set_defaults(f=cmd_serve);

    a = p.parse_args()
    a.f(a)
