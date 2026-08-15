"""A-5 · 검색기 자체 점검 (라벨 없이 돌리는 하한선 측정)

**주의: 이건 진짜 평가가 아니다.**

진짜 recall 은 라벨 시트의 `evidence_text` 를 `segment.find_evidence()` 로
구간에 붙여서 재야 한다(README_A2 참고). 그 라벨 시트가 아직 레포에 없다.

그래서 여기서는 **claim 자체가 원문에 거의 그대로 들어 있는 경우**만
골라서 정답으로 쓴다. `find_evidence(claim_text)` 가 exact/nospace 로 잡히면
그 claim 의 정답 구간을 아는 셈이다.

이 부분집합은 어휘가 일치하는 claim 들이라 **BM25 에 유리하고 dense 에 불리한
쪽으로 편향돼 있다.** 즉 여기서 나온 recall 은 dense 검색기의 하한선이다.
"이 정도는 최소한 맞힌다"를 확인하는 용도이지, 성능 보고용 숫자가 아니다.

라벨 시트가 들어오면 `--evidence <csv>` 로 진짜 정답을 넣어 다시 돌린다.

사용법
  python src/eval_retrieval.py --deck-id arts_01__claudecode
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from segment import Sentence, Span, find_evidence, normalize  # noqa: E402

CUTOFFS = (1, 5, 10, 20)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_gold(root: Path, doc_id: str, queries: list[dict],
              min_len: int) -> dict[str, set[str]]:
    """claim 이 원문에 그대로 있는 경우만 정답으로 인정한다.

    `find_evidence` 는 **첫 번째** 일치만 돌려준다. "카이로", "1961" 처럼
    짧은 claim 은 원문 여러 곳에 나오는데, 첫 곳만 정답으로 잡으면 다른 곳을
    맞힌 검색기가 틀린 것으로 집계된다. 그래서 exact 로 잡힌 claim 은
    **그 문자열을 담은 모든 구간**을 정답 집합에 넣는다.
    """
    sents = [Sentence(**r) for r in read_jsonl(root / "spans" / "sents" / f"{doc_id}.jsonl")]
    spans = [Span(**r) for r in read_jsonl(root / "spans" / "w3" / f"{doc_id}.jsonl")]
    norm_spans = [(sp.span_id, normalize(sp.text)) for sp in spans]

    gold: dict[str, set[str]] = {}
    for q in queries:
        text = q["query_text"]
        if len(text) < min_len:
            continue
        m = find_evidence(sents, spans, text)
        if m.method not in ("exact", "nospace") or not m.span_ids:
            continue
        ids = set(m.span_ids)
        needle = normalize(text)
        ids |= {sid for sid, body in norm_spans if needle in body}
        gold[q["claim_id"]] = ids
    return gold


def evaluate(runs: list[dict], gold: dict[str, set[str]]) -> dict:
    """정답 span 집합 중 **하나라도** top-k 에 있으면 hit.

    윈도우가 stride 1 로 겹치기 때문에 한 문장은 여러 구간에 들어간다.
    정답을 하나로 잡으면 recall 이 실제보다 낮게 나온다 (README_A2).
    """
    hits = {k: 0 for k in CUTOFFS}
    rr_sum = 0.0
    n = 0
    misses = []

    for run in runs:
        want = gold.get(run["claim_id"])
        if not want:
            continue
        n += 1
        first = None
        for r in run["results"]:
            span_id = r["passage_id"][len(run["doc_id"]) + 1:]
            if span_id in want:
                first = r["rank"]
                break
        if first is None:
            misses.append((run["claim_id"], run["query_text"]))
            continue
        rr_sum += 1.0 / first
        for k in CUTOFFS:
            if first <= k:
                hits[k] += 1

    return {
        "n": n,
        "recall": {k: (hits[k] / n if n else 0.0) for k in CUTOFFS},
        "mrr": (rr_sum / n if n else 0.0),
        "misses": misses,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck-id", required=True)
    ap.add_argument("--min-len", type=int, default=4,
                    help="이 길이 미만 claim 은 정답 후보에서 뺀다")
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()

    runs = read_jsonl(args.root / "retrieval" / "dense" / f"{args.deck_id}.jsonl")
    queries = read_jsonl(args.root / "retrieval" / "queries" / f"{args.deck_id}.jsonl")
    doc_id = queries[0]["doc_id"]

    gold = load_gold(args.root, doc_id, queries, args.min_len)
    res = evaluate(runs, gold)

    print(f"덱 {args.deck_id}  (doc_id={doc_id})")
    print(f"claim {len(queries)}개 중 원문에 그대로 있는 {res['n']}개로 하한선 측정\n")
    if not res["n"]:
        print("정답을 만들 수 있는 claim 이 없다.")
        return 0
    for k in CUTOFFS:
        print(f"  recall@{k:<2} {res['recall'][k]:.3f}")
    print(f"  MRR      {res['mrr']:.3f}")

    if res["misses"]:
        print(f"\n  top-20 안에도 못 찾은 {len(res['misses'])}개:")
        for cid, text in res["misses"][:5]:
            print(f"    {cid}  {text[:60]}")

    # 길이 구간별로 쪼개 본다 — dense 가 어디서 무너지는지가 BM25 담당자에게 필요한 정보
    print("\n  claim 길이별 top-1 cosine (정답 유무와 무관, 전체 claim 기준)")
    buckets = {"<10자": [], "10-29자": [], "30자+": []}
    q_by_id = {q["claim_id"]: q for q in queries}
    for run in runs:
        n_chars = len(q_by_id[run["claim_id"]]["query_text"])
        key = "<10자" if n_chars < 10 else ("10-29자" if n_chars < 30 else "30자+")
        buckets[key].append(run["results"][0]["score"])
    for key, scores in buckets.items():
        if scores:
            scores.sort()
            print(f"    {key:<8} n={len(scores):<4} 중앙값 {scores[len(scores) // 2]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
