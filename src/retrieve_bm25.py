from __future__ import annotations
"""A-4 · BM25 검색기 (희소 검색기)

claim 과 passage 를 형태소로 쪼갠 뒤 Okapi BM25 로 점수를 매겨, **같은 문서
안에서만** 상위 N 개를 뽑는다. A-5(dense_retrieve.py)와 정확히 같은 자리를
차지하는 짝이다 — 입력·출력 스키마를 맞춰서 RRF 가 `retriever` 필드만
보고 두 결과를 그대로 융합할 수 있게 한다 (README_A5.md "팀 계약" 참고).

    claims/*.jsonl 을 직접 읽지 않는다. retrieval/queries/{deck_id}.jsonl 을
    읽는다 — claim_id 가 build_queries.py 에서 이미 한 번만 부여됐기 때문에
    BM25 와 dense 가 같은 ID 를 쓰게 된다.

형태소 토큰화가 핵심이다. 공백으로만 나누면 "정책을/정책이/정책은" 이
전부 다른 토큰이 되어 색인에서 갈라진다. kiwi.tokenize() 의 form 을 쓰면
셋 다 "정책"(+조사)으로 갈라져서 "정책" 토큰으로 만난다 (README_A2.md 인계
사항). 조사·어미·구두점을 따로 걸러내지 않는다 — 그런 토큰은 거의 모든
passage 에 나타나므로 BM25 의 IDF 가 알아서 가중치를 낮춘다.

torch 를 요구하지 않는다. build_queries.py 와 같은 이유다 — dense 담당자가
쓰는 임베딩 스택을 이쪽에서 설치할 필요가 없어야 한다.

사용법
  python src/retrieve_bm25.py --deck-id arts_01__claudecode
  python src/retrieve_bm25.py --all --top-n 20
  python src/retrieve_bm25.py --deck-id arts_01__claudecode --window w5
  python src/retrieve_bm25.py --deck-id arts_01__claudecode --show 5   # 눈으로 확인
"""


import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

DEFAULT_TOP_N = 20          # dense_retrieve.py 와 맞춘다. RRF 여유분.
DEFAULT_WINDOW = "w3"
DEFAULT_K1 = 1.5             # 항 빈도 포화 속도 (Okapi 관용값)
DEFAULT_B = 0.75             # 문서 길이 정규화 강도 (Okapi 관용값)

RETRIEVER_NAME = "bm25"


# ─────────────────────────────────────────────────────────────
# 입출력 — dense_retrieve.py 와 각자 복사본을 둔다 (모듈 상단 import 만으로
# torch/sentence-transformers 가 딸려오지 않게 하려고 일부러 공유하지 않는다.
# build_passages.py / build_queries.py 도 같은 이유로 각자 복사본을 둔다).
# ─────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def passage_path(root: Path, doc_id: str, window: str) -> Path:
    name = f"{doc_id}.jsonl" if window == DEFAULT_WINDOW else f"{doc_id}.{window}.jsonl"
    return root / "passages" / name


# ─────────────────────────────────────────────────────────────
# 토큰화
# ─────────────────────────────────────────────────────────────

_KIWI = None


def kiwi():
    """kiwi 는 초기화가 느리다. 덱 여러 개를 돌 때 한 번만 올린다."""
    global _KIWI
    if _KIWI is None:
        from kiwipiepy import Kiwi
        _KIWI = Kiwi()
    return _KIWI


def tokenize(text: str) -> list[str]:
    """형태소 form 리스트로 바꾼다. 품사 필터링은 하지 않는다 — 위 docstring 참고."""
    return [t.form for t in kiwi().tokenize(text) if t.form.strip()]


# ─────────────────────────────────────────────────────────────
# Okapi BM25
# ─────────────────────────────────────────────────────────────

class BM25Index:
    """문서(passage) 집합 하나에 대한 BM25 색인.

    문서 간 검색을 하지 않으므로(같은 doc_id 안에서만 검색) 덱 하나를 처리할
    때마다 passage pool 크기의 색인을 새로 만든다. passage pool 이 40~200개
    수준이라 벡터 DB 나 역색인 라이브러리 없이 이 정도로 충분하다.
    """

    def __init__(self, passages: list[dict], k1: float = DEFAULT_K1, b: float = DEFAULT_B):
        self.k1 = k1
        self.b = b
        self.passage_ids = [p["passage_id"] for p in passages]
        self.doc_tokens = [tokenize(p["text"]) for p in passages]
        self.doc_len = np.array([len(toks) for toks in self.doc_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0

        self.tf: list[Counter[str]] = [Counter(toks) for toks in self.doc_tokens]
        n_docs = len(passages)
        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        # Robertson-Sparck Jones IDF. 음수가 나올 수 있는 원형 대신
        # +1 스무딩된 형태를 쓴다 — 극단적으로 흔한 토큰도 완전히 0 이하로
        # 떨어지지 않아 점수 안정성이 좋다 (Elasticsearch/Lucene 관용값).
        self.idf: dict[str, float] = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query_tokens: list[str]) -> np.ndarray:
        """passage 개수만큼의 점수 배열. query 안의 중복 토큰도 그대로 반영한다."""
        scores = np.zeros(len(self.doc_tokens), dtype=np.float32)
        if not query_tokens or self.avgdl == 0.0:
            return scores
        for term, q_freq in Counter(query_tokens).items():
            idf = self.idf.get(term)
            if idf is None or idf <= 0:
                continue  # 색인에 없는 토큰(passage 어디에도 없음) — 기여 0
            for i, tf_i in enumerate(self.tf):
                f = tf_i.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * q_freq * (f * (self.k1 + 1)) / denom
        return scores


# ─────────────────────────────────────────────────────────────
# 검색
# ─────────────────────────────────────────────────────────────

def retrieve(queries: list[dict], passages: list[dict], index: BM25Index,
             top_n: int) -> list[dict]:
    k = min(top_n, len(passages))
    out = []
    for q in queries:
        q_tokens = tokenize(q["query_text"])
        scores = index.score(q_tokens)
        order = np.argsort(-scores, kind="stable")[:k]
        out.append({
            "claim_id": q["claim_id"],
            "deck_id": q["deck_id"],
            "doc_id": q["doc_id"],
            "retriever": RETRIEVER_NAME,
            "model": f"kiwi+okapi_bm25(k1={index.k1},b={index.b})",
            "window": passages[0]["window"],
            "query_text": q["query_text"],
            "results": [
                {
                    "rank": r,
                    "passage_id": passages[j]["passage_id"],
                    "score": round(float(scores[j]), 6),
                    "sent_ids": passages[j]["sent_ids"],
                }
                for r, j in enumerate(order, start=1)
            ],
        })
    return out


def run_deck(root: Path, deck_id: str, top_n: int, window: str, k1: float,
             b: float, show: int) -> int:
    q_path = root / "retrieval" / "queries" / f"{deck_id}.jsonl"
    if not q_path.exists():
        print(f"SKIP {deck_id}: {q_path} 가 없다. 먼저 build_queries.py 를 돌려라.",
              file=sys.stderr)
        return 1
    queries = read_jsonl(q_path)
    doc_ids = {q["doc_id"] for q in queries}
    if len(doc_ids) != 1:
        print(f"FAIL {deck_id}: 한 덱에 doc_id 가 여러 개다 {doc_ids}", file=sys.stderr)
        return 1
    doc_id = doc_ids.pop()

    p_path = passage_path(root, doc_id, window)
    if not p_path.exists():
        print(f"SKIP {deck_id}: {p_path} 가 없다 (doc_id={doc_id}). "
              f"원문 정제가 아직 안 된 문서다.", file=sys.stderr)
        return 1
    passages = read_jsonl(p_path)

    index = BM25Index(passages, k1=k1, b=b)
    rows = retrieve(queries, passages, index, top_n)

    name = f"{deck_id}.jsonl" if window == DEFAULT_WINDOW else f"{deck_id}.{window}.jsonl"
    out = root / "retrieval" / "bm25" / name
    write_jsonl(out, rows)

    top1 = [r["results"][0]["score"] for r in rows if r["results"]]
    zero_hits = sum(1 for r in rows if not r["results"] or r["results"][0]["score"] == 0)
    print(f"OK   {deck_id}: {len(rows)} claims x {len(passages)} passages "
          f"-> {out}")
    if top1:
        print(f"     top-1 BM25 중앙값 {np.median(top1):.4f}  "
              f"최소 {min(top1):.4f}  최대 {max(top1):.4f}")
    print(f"     어휘가 하나도 안 겹치는 claim {zero_hits}개 "
          f"({zero_hits / len(rows):.0%}) — dense 가 메워야 할 구간")

    for row in rows[:show]:
        print(f"\n  [{row['claim_id']}] {row['query_text'][:60]}")
        by_id = {p["passage_id"]: p for p in passages}
        for hit in row["results"][:3]:
            text = by_id[hit["passage_id"]]["text"]
            print(f"    {hit['rank']}. {hit['passage_id']} ({hit['score']:.4f}) {text[:70]}...")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck-id")
    ap.add_argument("--all", action="store_true", help="retrieval/queries 안의 모든 덱")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--window", default=DEFAULT_WINDOW, choices=["w3", "w5"])
    ap.add_argument("--k1", type=float, default=DEFAULT_K1)
    ap.add_argument("--b", type=float, default=DEFAULT_B)
    ap.add_argument("--show", type=int, default=0, help="상위 결과를 N개 claim 만큼 찍어본다")
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()

    if args.all:
        deck_ids = sorted(p.stem for p in (args.root / "retrieval" / "queries").glob("*.jsonl"))
    elif args.deck_id:
        deck_ids = [args.deck_id]
    else:
        ap.error("--deck-id 또는 --all 이 필요하다")

    skipped = 0
    for deck_id in deck_ids:
        skipped += run_deck(args.root, deck_id, args.top_n, args.window,
                            args.k1, args.b, args.show)
    if skipped:
        print(f"\n{skipped}개 덱을 건너뛰었다 (passage pool 없음).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
