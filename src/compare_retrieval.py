#!/usr/bin/env python3
"""B-1 · 검색 방식 비교 — BM25 vs 임베딩 vs 하이브리드(RRF)

`eval_retrieval.py` 는 "claim 이 원문에 그대로 있는 경우"만 정답으로 써서
하한선만 잰다. 그 파일 주석대로 **라벨 시트가 들어오면 진짜 정답으로 다시
돌려야** 하고, 이 스크립트가 그 자리다.

정답은 어노테이터가 원문에서 복붙한 `evidence_text` 다.
`segment.find_evidence()` 로 구간에 붙여서 정답 span 집합을 만든다.

    python src/compare_retrieval.py \
        --root <레포> --deck-id arts_01__claudecode \
        --annotations annotation/annotations_long.csv

윈도우가 stride 1 로 겹치므로 정답은 **집합**이다. 하나라도 top-k 안에
들어오면 hit (README_A2 "정답이 여러 개인 게 정상이다").
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

CUTOFFS = (1, 3, 5, 10)
RRF_K = 60
# 하이브리드 가중치 (BM25, 임베딩). 0.5/0.5 는 가중치 없는 RRF 와 순위가 같다.
HYBRID_WEIGHTS = [(0.3, 0.7), (0.5, 0.5), (0.7, 0.3)]
# 근거 문장이 있어야 정답을 만들 수 있다. Benign·무근거는 애초에 근거가 없다.
GOLD_LABELS = ("근거 있음", "모순")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


# 복붙이 완벽하지 않다. 실제로 나온 두 패턴만 걷어낸다.
_ELLIPSIS = re.compile(r"\s*(?:\.{2,}|…|⋯)\s*")          # "앞부분... 뒷부분"
_TAIL = re.compile(r"\s*[（(]\s*p\.?\s*\d+[^)）]*[)）]\s*$")  # "(p.131, 소제목)"
_COMMENT = re.compile(r"\s*[—–]\s*[^—–]+$")                # "본문 — 내 설명"


def evidence_parts(raw: str) -> list[str]:
    """근거 문장 하나를 원문에서 찾을 수 있는 조각들로 만든다.

    가이드는 "원문 문장 그대로 복붙"을 요구했지만 실제 시트에는
    ``...`` 로 중간을 생략하거나 ``(p.131, 소제목)`` 같은 메모를 붙인
    행이 많다. 조각으로 쪼개면 그 중 하나는 원문에 그대로 있다.
    """
    out: list[str] = []
    for chunk in raw.split(" || "):
        for part in _ELLIPSIS.split(chunk):
            part = _COMMENT.sub("", _TAIL.sub("", part)).strip()
            if len(part) >= 6:
                out.append(part)
    return out or [raw.strip()]


# ─────────────────────────────────────────────────────────────
# 정답 만들기
# ─────────────────────────────────────────────────────────────
def _bigrams(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


def overlap_spans(spans, evidence: str, floor: float = 0.55) -> list[str]:
    """문자 2-gram 겹침으로 근거가 어느 구간인지 고른다.

    `find_evidence` 는 근거가 원문에 **연속으로** 들어 있다고 본다. 그런데
    어노테이터는 PDF 뷰어에서 복붙했고, 2단 조판 논문은 뷰어가 좌우 단을
    한 줄에 섞어서 준다. 그러면 같은 내용인데도 연속 부분문자열이 아니라서
    exact/nospace/fuzzy 가 전부 실패한다.

    그래서 순서를 버리고 문자 2-gram 집합의 포함률로 본다. 근거의 2-gram 중
    구간 안에 들어 있는 비율이 floor 를 넘으면 그 구간을 정답으로 인정한다.
    정확도는 떨어지므로 이 경로로 만든 정답은 따로 센다.
    """
    ev = _bigrams(evidence)
    if len(ev) < 8:
        return []
    scored = []
    for sp in spans:
        got = len(ev & _bigrams(sp.text)) / len(ev)
        if got >= floor:
            scored.append((got, sp.span_id))
    if not scored:
        return []
    best = max(s for s, _ in scored)
    # 최고점에 가까운 구간은 다 정답에 넣는다 — 윈도우가 겹치므로 정상이다
    return [sid for s, sid in scored if s >= best - 0.05]


def build_gold(root: Path, doc_id: str, queries: list[dict],
               ann_rows: list[dict]) -> tuple[dict[str, dict[str, set[str]]], dict[str, int]]:
    """(어노테이터 -> {claim_id: 정답 span 집합}, 실패 사유별 개수)."""
    sys.path.insert(0, str(root / "src"))
    from segment import Sentence, Span, find_evidence  # noqa: E402

    sents = [Sentence(**r) for r in read_jsonl(root / "spans" / "sents" / f"{doc_id}.jsonl")]
    spans = [Span(**r) for r in read_jsonl(root / "spans" / "w3" / f"{doc_id}.jsonl")]

    by_text: dict[str, str] = {}
    for q in queries:
        by_text.setdefault(norm_key(q["query_text"]), q["claim_id"])

    gold: dict[str, dict[str, set[str]]] = defaultdict(dict)
    why: dict[str, int] = defaultdict(int)
    cache: dict[str, tuple[list[str], str]] = {}

    for r in ann_rows:
        if r["label"] not in GOLD_LABELS:
            why["근거 없는 라벨(Benign·무근거·공란)"] += 1
            continue
        ev = (r.get("evidence_text") or "").strip()
        if not ev:
            why["근거 문장 비어 있음"] += 1
            continue
        cid = by_text.get(norm_key(r["claim_text"]))
        if cid is None:
            why["claim 이 현재 분할기 출력에 없음"] += 1
            continue
        if ev not in cache:
            found: list[str] = []
            for part in evidence_parts(ev):
                found += find_evidence(sents, spans, part).span_ids
            how = "연속 일치"
            if not found:
                found = overlap_spans(spans, ev)
                how = "2-gram 겹침" if found else ""
            cache[ev] = (found, how)
        ids, how = cache[ev]
        if not ids:
            why["근거 문장을 원문에서 못 찾음"] += 1
            continue
        gold[r["annotator"]][cid] = set(ids)
        why[f"정답 생성 ({how})"] += 1
    return gold, dict(why)


# ─────────────────────────────────────────────────────────────
# 검색 결과 읽기 · 융합
# ─────────────────────────────────────────────────────────────
def load_runs(path: Path) -> dict[str, list[str]]:
    """claim_id -> 순위대로 나열한 span_id."""
    out: dict[str, list[str]] = {}
    for run in read_jsonl(path):
        pre = len(run["doc_id"]) + 1
        ranked = sorted(run["results"], key=lambda r: r["rank"])
        out[run["claim_id"]] = [r["passage_id"][pre:] for r in ranked]
    return out


def rrf(runs: list[dict[str, list[str]]], weights: list[float] | None = None,
        k: int = RRF_K) -> dict[str, list[str]]:
    """가중 Reciprocal Rank Fusion — 점수가 아니라 순위만 쓴다.

        score(구간) = Σ  wᵢ / (k + 그 검색기에서의 등수)

    BM25 점수(0~48)와 cosine(0.71~0.90)은 스케일이 달라 직접 더하면 BM25 가
    결과를 지배한다. 등수로 바꾸면 두 검색기가 대등해진다.

    k 는 1등의 힘을 눌러주는 상수다. k=60 이면 1등 1/61, 2등 1/62 로 거의
    같아서, **한쪽이 1등으로 꼽은 것**보다 **양쪽이 모두 상위권에 올린 것**이
    이긴다. 한쪽이 크게 틀려도 다른 쪽이 받쳐준다.

    가중치는 상대적으로만 작동한다. (0.5, 0.5) 는 (1, 1) 과 순위가 같다.
    """
    if weights is None:
        weights = [1.0] * len(runs)
    out: dict[str, list[str]] = {}
    for cid in set().union(*(r.keys() for r in runs)):
        score: dict[str, float] = defaultdict(float)
        for w, run in zip(weights, runs):
            for rank, sid in enumerate(run.get(cid, []), start=1):
                score[sid] += w / (k + rank)
        out[cid] = [s for s, _ in sorted(score.items(), key=lambda x: -x[1])]
    return out


def score(runs: dict[str, list[str]], gold: dict[str, set[str]]) -> dict[str, Any]:
    hits = dict.fromkeys(CUTOFFS, 0)
    rr = 0.0
    n = 0
    first_rank: dict[str, int | None] = {}
    for cid, want in gold.items():
        ranked = runs.get(cid)
        if ranked is None:
            continue
        n += 1
        first = next((i for i, sid in enumerate(ranked, 1) if sid in want), None)
        first_rank[cid] = first
        if first is None:
            continue
        rr += 1.0 / first
        for c in CUTOFFS:
            if first <= c:
                hits[c] += 1
    return {"n": n,
            "recall": {c: hits[c] / n if n else 0.0 for c in CUTOFFS},
            "mrr": rr / n if n else 0.0,
            "first_rank": first_rank}


def hit_set(res: dict[str, Any], k: int) -> set[str]:
    """top-k 안에 정답이 들어온 claim 들."""
    return {cid for cid, r in res["first_rank"].items() if r is not None and r <= k}


def table(title: str, results: dict[str, dict], n: int) -> str:
    head = "  ".join(f"R@{c}" for c in CUTOFFS)
    out = [f"\n{title}  (정답 있는 claim {n}개)",
           f"  {'검색 방식':<12} {head}   MRR",
           f"  {'-' * 46}"]
    for name, r in results.items():
        cells = "  ".join(f"{r['recall'][c]:.3f}" for c in CUTOFFS)
        out.append(f"  {name:<12} {cells}  {r['mrr']:.3f}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="레포 루트")
    ap.add_argument("--deck-id", required=True)
    ap.add_argument("--annotations", type=Path, required=True,
                    help="normalize_annotation.py 가 만든 annotations_long.csv")
    ap.add_argument("--out", type=Path, help="결과 마크다운 저장 위치")
    ap.add_argument("--dump", type=Path,
                    help="claim 별로 정답 구간과 검색 결과를 나란히 적은 CSV 저장 위치")
    ap.add_argument("--metrics-dir", type=Path,
                    help="덱별 지표를 JSON 으로 남길 폴더 (summarize_b1.py 입력)")
    args = ap.parse_args()

    root = args.root
    queries = read_jsonl(root / "retrieval" / "queries" / f"{args.deck_id}.jsonl")
    doc_id = queries[0]["doc_id"]

    with args.annotations.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["deck_id"] == args.deck_id]
    if not rows:
        raise SystemExit(f"{args.deck_id} 에 해당하는 어노테이션 행이 없다")

    gold, why = build_gold(root, doc_id, queries, rows)

    runs = {}
    for name, sub in (("BM25", "bm25"), ("임베딩", "dense")):
        p = root / "retrieval" / sub / f"{args.deck_id}.jsonl"
        if p.exists():
            runs[name] = load_runs(p)
    if len(runs) < 2:
        raise SystemExit(f"BM25·임베딩 결과가 둘 다 필요하다: {sorted(runs)}")

    # 가중치를 바꾼 하이브리드 3버전. (BM25 비중, 임베딩 비중)
    base = [runs["BM25"], runs["임베딩"]]
    for wb, wd in HYBRID_WEIGHTS:
        runs[f"하이브리드 {wb:.1f}/{wd:.1f}"] = rrf(base, [wb, wd])

    lines: list[str] = []
    add = lines.append
    add(f"# 검색 방식 비교 — {args.deck_id}")
    add(f"\n문서 `{doc_id}` · claim {len(queries)}개 · 구간 w3 · RRF k={RRF_K}")
    add(f"\n정답은 어노테이터가 원문에서 복붙한 `evidence_text` 다.\n")

    add("## 정답으로 쓸 수 있었던 라벨\n")
    add("| 처리 | 행 |")
    add("| --- | ---: |")
    for k, v in sorted(why.items(), key=lambda x: -x[1]):
        add(f"| {k} | {v} |")

    # 어노테이터별
    add("\n## 어노테이터별\n")
    add("| 어노테이터 | 정답 claim | " + " | ".join(f"R@{c}" for c in CUTOFFS) + " | MRR | 방식 |")
    add("| --- | ---: | " + " | ".join(["---:"] * len(CUTOFFS)) + " | ---: | --- |")
    per: dict[str, dict[str, dict]] = {}
    for who in sorted(gold):
        per[who] = {name: score(r, gold[who]) for name, r in runs.items()}
        for name, res in per[who].items():
            cells = " | ".join(f"{res['recall'][c]:.3f}" for c in CUTOFFS)
            add(f"| {who} | {res['n']} | {cells} | {res['mrr']:.3f} | {name} |")

    # 통합 — 어노테이터들의 정답을 합집합으로
    pooled: dict[str, set[str]] = defaultdict(set)
    for who in gold:
        for cid, ids in gold[who].items():
            pooled[cid] |= ids
    combined = {name: score(r, pooled) for name, r in runs.items()}

    add("\n## 통합 (어노테이터 정답 합집합)\n")
    add("| 검색 방식 | " + " | ".join(f"R@{c}" for c in CUTOFFS) + " | MRR |")
    add("| --- | " + " | ".join(["---:"] * len(CUTOFFS)) + " | ---: |")
    for name, r in combined.items():
        cells = " | ".join(f"{r['recall'][c]:.3f}" for c in CUTOFFS)
        add(f"| {name} | {cells} | {r['mrr']:.3f} |")

    q_text = {q["claim_id"]: q["query_text"] for q in queries}

    if args.dump:
        span_text = {r["span_id"]: r["text"] for r in
                     read_jsonl(root / "spans" / "w3" / f"{doc_id}.jsonl")}
        ev_of: dict[str, list[str]] = defaultdict(list)
        by_text = {norm_key(q["query_text"]): q["claim_id"] for q in queries}
        for r in rows:
            cid = by_text.get(norm_key(r["claim_text"]))
            if cid and r.get("evidence_text", "").strip():
                ev_of[cid].append(f"[{r['annotator']}] {r['evidence_text'].strip()}")

        args.dump.parent.mkdir(parents=True, exist_ok=True)
        with args.dump.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["claim_id", "claim_text", "정답구간", "정답구간_본문",
                        "어노테이터_근거",
                        *[f"{n}_1위" for n in runs], *[f"{n}_1위_본문" for n in runs],
                        *[f"{n}_정답순위" for n in runs]])
            for cid in sorted(pooled):
                want = pooled[cid]
                top1 = {n: (runs[n].get(cid) or [""])[0] for n in runs}
                rank = {n: combined[n]["first_rank"].get(cid) for n in runs}
                w.writerow([
                    cid, q_text.get(cid, ""),
                    " ".join(sorted(want)),
                    " ⏸ ".join(span_text.get(s, "") for s in sorted(want)[:1]),
                    " ⏸ ".join(ev_of.get(cid, [])),
                    *[top1[n] for n in runs],
                    *[span_text.get(top1[n], "") for n in runs],
                    *[("못 찾음" if rank[n] is None else rank[n]) for n in runs],
                ])
        print(f"claim별 상세: {args.dump}")

    # claim 길이별 — 짧은 claim 에서 임베딩이 무너진다는 가설을 확인한다
    add("\n## claim 길이별 R@1 (통합 정답)\n")
    add("| 길이 | claim | " + " | ".join(runs) + " |")
    add("| --- | ---: | " + " | ".join(["---:"] * len(runs)) + " |")
    buckets = (("10자 미만", 0, 10), ("10-29자", 10, 30), ("30자 이상", 30, 10 ** 6))
    for name, lo, hi in buckets:
        ids = {c for c in pooled if lo <= len(q_text.get(c, "")) < hi}
        if not ids:
            continue
        cells = []
        for r in combined.values():
            got = len(hit_set(r, 1) & ids)
            cells.append(f"{got / len(ids):.3f}")
        add(f"| {name} | {len(ids)} | " + " | ".join(cells) + " |")

    # 어디서 갈리는지 — top-1 기준으로 한쪽만 맞힌 claim
    b, d = hit_set(combined["BM25"], 1), hit_set(combined["임베딩"], 1)
    add("\n## 서로 다른 곳 (top-1 기준)\n")
    add(f"- BM25만 맞힘 {len(b - d)}개 · 임베딩만 맞힘 {len(d - b)}개 · "
        f"둘 다 맞힘 {len(b & d)}개 · 둘 다 놓침 {len(set(pooled) - b - d)}개\n")
    for label, ids in (("BM25만 맞힌 claim", b - d), ("임베딩만 맞힌 claim", d - b)):
        if ids:
            add(f"**{label}**\n")
            for cid in sorted(ids)[:10]:
                add(f"- `{cid}` {q_text.get(cid, '')[:60]}")
            add("")

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\n저장: {args.out}")

    if args.metrics_dir:
        # 덱을 가로지르는 요약표는 summarize_b1.py 가 이 파일들을 모아 만든다
        args.metrics_dir.mkdir(parents=True, exist_ok=True)
        (args.metrics_dir / f"{args.deck_id}.json").write_text(json.dumps({
            "deck_id": args.deck_id,
            "doc_id": doc_id,
            "n_claims": len(queries),
            "n_gold": len(pooled),
            "gold_source": why,
            "methods": {name: {"recall": {str(c): r["recall"][c] for c in CUTOFFS},
                               "mrr": r["mrr"], "n": r["n"]}
                        for name, r in combined.items()},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
