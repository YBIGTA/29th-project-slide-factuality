"""A-5 · 임베딩 검색기 (dense retriever)

multilingual-e5 로 passage 와 claim 을 같은 공간에 넣고, **같은 문서 안에서만**
cosine 유사도 상위 N 개를 뽑는다.

    passage:  "passage: {구간 텍스트}"
    claim:    "query: {claim 텍스트}"

e5 계열은 이 접두사를 붙여 학습됐다. 빼면 성능이 눈에 띄게 떨어지고,
둘 다 `query:` 로 넣어도 마찬가지다. README_A2 가 지정한 규약이기도 하다.

출력은 rank 를 담고 있어서 BM25 결과와 그대로 RRF 융합할 수 있다.
score(cosine)도 같이 싣지만 **RRF 는 rank 만 쓴다.** BM25 점수와 cosine 은
스케일이 달라서 직접 더하면 안 된다.

사용법
  python src/dense_retrieve.py --deck-id arts_01__claudecode
  python src/dense_retrieve.py --all --top-n 20
  python src/dense_retrieve.py --deck-id arts_01__claudecode --window w5
  python src/dense_retrieve.py --deck-id arts_01__claudecode --show 5   # 눈으로 확인
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_TOP_N = 20          # RRF 여유분. 판별 모델에는 융합 후 top-5 가 간다.
DEFAULT_WINDOW = "w3"


# ─────────────────────────────────────────────────────────────
# 입출력
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
# 인코딩
# ─────────────────────────────────────────────────────────────

_MODEL = None


def load_model(name: str, device: str | None):
    """모델은 한 번만 올린다. 덱 여러 개를 돌 때 재사용된다."""
    global _MODEL
    if _MODEL is None or _MODEL[0] != name:
        from sentence_transformers import SentenceTransformer
        _MODEL = (name, SentenceTransformer(name, device=device))
    return _MODEL[1]


def encode(model, texts: list[str], prefix: str, batch_size: int = 16) -> np.ndarray:
    """e5 접두사를 붙여 인코딩하고 L2 정규화한다.

    정규화해두면 내적이 곧 cosine 이라 뒤에서 행렬곱 한 번으로 끝난다.
    """
    tagged = [f"{prefix}{t}" for t in texts]
    vecs = model.encode(
        tagged,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def _fingerprint(texts: list[str], model_name: str) -> str:
    """passage 내용이 바뀌면 캐시를 버려야 한다."""
    h = hashlib.sha256(model_name.encode("utf-8"))
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:16]


def encode_passages_cached(model, model_name: str, passages: list[dict],
                           cache_dir: Path, doc_id: str, window: str) -> np.ndarray:
    texts = [p["text"] for p in passages]
    fp = _fingerprint(texts, model_name)
    slug = model_name.replace("/", "__")
    path = cache_dir / f"{doc_id}__{window}__{slug}__{fp}.npy"
    if path.exists():
        return np.load(path)
    vecs = encode(model, texts, "passage: ")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs


# ─────────────────────────────────────────────────────────────
# 검색
# ─────────────────────────────────────────────────────────────

def retrieve(queries: list[dict], passages: list[dict], q_vecs: np.ndarray,
             p_vecs: np.ndarray, top_n: int, model_name: str) -> list[dict]:
    sims = q_vecs @ p_vecs.T                      # (claim 수, passage 수) cosine
    k = min(top_n, len(passages))
    # 상위 k 개만 뽑고 그 안에서 정렬한다. 전체 정렬보다 싸다.
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]

    out = []
    for i, q in enumerate(queries):
        idx = part[i][np.argsort(-sims[i, part[i]])]
        out.append({
            "claim_id": q["claim_id"],
            "deck_id": q["deck_id"],
            "doc_id": q["doc_id"],
            "retriever": "dense_e5",
            "model": model_name,
            "window": passages[0]["window"],
            "query_text": q["query_text"],
            "results": [
                {
                    "rank": r,
                    "passage_id": passages[j]["passage_id"],
                    "score": round(float(sims[i, j]), 6),
                    "sent_ids": passages[j]["sent_ids"],
                }
                for r, j in enumerate(idx, start=1)
            ],
        })
    return out


def run_deck(root: Path, deck_id: str, model_name: str, device: str | None,
             top_n: int, window: str, show: int) -> int:
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

    model = load_model(model_name, device)
    p_vecs = encode_passages_cached(model, model_name, passages,
                                    root / ".cache" / "emb", doc_id, window)
    q_vecs = encode(model, [q["query_text"] for q in queries], "query: ")

    rows = retrieve(queries, passages, q_vecs, p_vecs, top_n, model_name)
    name = f"{deck_id}.jsonl" if window == DEFAULT_WINDOW else f"{deck_id}.{window}.jsonl"
    out = root / "retrieval" / "dense" / name
    write_jsonl(out, rows)

    top1 = [r["results"][0]["score"] for r in rows]
    print(f"OK   {deck_id}: {len(rows)} claims x {len(passages)} passages "
          f"-> {out}")
    print(f"     top-1 cosine 중앙값 {np.median(top1):.4f}  "
          f"최소 {min(top1):.4f}  최대 {max(top1):.4f}")

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
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None, help="mps | cuda | cpu (기본: 자동)")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--window", default=DEFAULT_WINDOW, choices=["w3", "w5"])
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
        skipped += run_deck(args.root, deck_id, args.model, args.device,
                            args.top_n, args.window, args.show)
    if skipped:
        print(f"\n{skipped}개 덱을 건너뛰었다 (passage pool 없음).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
