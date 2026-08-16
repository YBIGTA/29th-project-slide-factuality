#!/usr/bin/env python3
"""B-1 · 덱을 가로지르는 검색 방식 비교표를 만든다.

  python src/summarize_b1.py results/_metrics -o results/retrieval_summary.md

`compare_retrieval.py --metrics-dir` 가 덱마다 남긴 JSON 을 모아
한 표로 만든다. 발표에 올릴 숫자는 이 파일 하나만 보면 된다.

덱마다 정답 품질이 달라서 단순 평균은 오해를 부른다. 그래서
**덱별 표 + 정답이 믿을 만한 덱만 따로** 를 같이 낸다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CUTOFFS = ("1", "3", "5", "10")


def load(metrics_dir: Path) -> list[dict]:
    out = [json.loads(p.read_text(encoding="utf-8"))
           for p in sorted(metrics_dir.glob("*.json"))]
    if not out:
        raise SystemExit(f"{metrics_dir} 에 JSON 이 없다")
    return out


def clean_ratio(m: dict) -> float:
    """정답 중 '원문에 그대로 있어서' 만들어진 비율. 측정 신뢰도의 대리 지표."""
    src = m.get("gold_source", {})
    exact = sum(v for k, v in src.items() if "연속" in k)
    total = sum(v for k, v in src.items() if k.startswith("정답 생성"))
    return exact / total if total else 0.0


def table(rows: list[dict], methods: list[str], cut: str) -> list[str]:
    out = ["| 덱 | 정답 claim | " + " | ".join(methods) + " |",
           "| --- | ---: | " + " | ".join(["---:"] * len(methods)) + " |"]
    for m in rows:
        best = max(m["methods"][x]["recall"][cut] for x in methods)
        cells = []
        for x in methods:
            v = m["methods"][x]["recall"][cut]
            cells.append(f"**{v:.3f}**" if v == best else f"{v:.3f}")
        out.append(f"| {m['deck_id']} | {m['n_gold']} | " + " | ".join(cells) + " |")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    ap.add_argument("--trust-floor", type=float, default=0.5,
                    help="정답의 이 비율 이상이 원문 그대로여야 '신뢰 가능'으로 본다")
    args = ap.parse_args()

    rows = load(args.metrics_dir)
    methods = list(rows[0]["methods"])

    lines: list[str] = ["# B-1 · 검색 방식 비교", ""]
    lines.append(f"덱 {len(rows)}개 · 방식 {len(methods)}가지 · 구간 w3 · RRF k=60")
    lines.append("")
    lines.append("하이브리드 뒤 숫자는 (BM25 비중 / 임베딩 비중)이다. "
                 "가중치는 상대값이라 0.5/0.5 는 가중치 없는 RRF 와 같다.")
    lines.append("")

    for cut, title in (("5", "R@5 — 후보 5개 안에 근거가 들어올 확률"),
                       ("1", "R@1 — 1등이 맞을 확률")):
        lines += [f"## {title}", ""] + table(rows, methods, cut) + [""]

    lines += ["## MRR", "",
              "| 덱 | " + " | ".join(methods) + " |",
              "| --- | " + " | ".join(["---:"] * len(methods)) + " |"]
    for m in rows:
        best = max(m["methods"][x]["mrr"] for x in methods)
        cells = [(f"**{m['methods'][x]['mrr']:.3f}**"
                  if m["methods"][x]["mrr"] == best else f"{m['methods'][x]['mrr']:.3f}")
                 for x in methods]
        lines.append(f"| {m['deck_id']} | " + " | ".join(cells) + " |")
    lines.append("")

    # 정답 품질 — 이걸 안 보이면 위 표를 잘못 읽는다
    lines += ["## 정답 품질", "",
              "`원문 그대로` 는 어노테이터의 근거 문장이 정제본에 그대로 있어서 "
              "구간을 정확히 짚을 수 있었던 경우다. `추정` 은 2-gram 겹침으로 "
              "가장 비슷한 구간을 고른 것이라 **틀릴 수 있다.**", "",
              "| 덱 | 원문 그대로 | 추정 | 못 찾음 | 신뢰도 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    trusted = []
    for m in rows:
        src = m.get("gold_source", {})
        exact = sum(v for k, v in src.items() if "연속" in k)
        approx = sum(v for k, v in src.items() if "2-gram" in k)
        miss = sum(v for k, v in src.items() if "못 찾음" in k)
        r = clean_ratio(m)
        if r >= args.trust_floor:
            trusted.append(m)
        lines.append(f"| {m['deck_id']} | {exact} | {approx} | {miss} | {r:.0%} |")
    lines.append("")

    if trusted and len(trusted) < len(rows):
        lines += [f"## 정답이 믿을 만한 덱만 (원문 그대로 {args.trust_floor:.0%} 이상)", "",
                  "위 표에서 신뢰도가 낮은 덱은 검색기 성능이 아니라 정답 잡음을 재고 있다.",
                  ""] + table(trusted, methods, "5") + [""]

    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"\n저장: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
