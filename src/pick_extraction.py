#!/usr/bin/env python3
"""원문마다 1단/2단 추출을 둘 다 해보고 나은 쪽을 남긴다.

  python src/pick_extraction.py docs/raw/*.pdf
  python src/pick_extraction.py docs/raw/bio_02.pdf --doc-id bio_02

2단 조판 판정을 좌표로 추론해봤지만 표·수식·전각 글꼴 때문에 계속 틀렸다.
그래서 추론하지 않고 **두 방식으로 다 뽑아 점수를 재고 이긴 쪽을 쓴다.**

점수는 `문장이 종결어미로 제대로 끝나는 비율`이다. 이게 텍스트 손상을 직접
드러낸다 — 2단이 섞이면 문장이 중간에 끊기고, 1단을 억지로 반 가르면 모든
줄이 두 동강 나서, 어느 쪽이든 이 비율이 떨어진다.

    arts_01  1단 97.0%  2단 71.2%   -> 1단
    bio_02   1단 54.1%  2단 70.3%   -> 2단

이겨도 100% 가 아니다. 남은 손상은 사람이 review 파일과 본문을 보고 고쳐야
한다. 이 도구는 그 출발점을 가장 좋은 상태로 만들어줄 뿐이다.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pdf_to_text import pdf_to_text          # noqa: E402
from segment import split_sentences          # noqa: E402

# `~했다.` `~이다.` `~하였음.` `~가능함` 처럼 문장이 제대로 닫힌 모양
_ENDER = re.compile(r"(다|음|임|함|것|요|까|랐|셨|이다)[.)]?[\"”')\]]*$")


def quality(text: str) -> tuple[float, int, float]:
    """(종결어미로 끝나는 문장 비율, 문장 수, 평균 길이)."""
    body = [s for s in split_sentences(text) if not s.is_heading]
    if not body:
        return 0.0, 0, 0.0
    ok = sum(1 for s in body if _ENDER.search(s.text.strip()))
    return ok / len(body), len(body), sum(len(s.text) for s in body) / len(body)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("docs/clean"))
    ap.add_argument("--doc-id", help="PDF 하나만 줄 때 doc_id 를 직접 지정")
    ap.add_argument("--dry-run", action="store_true", help="점수만 보고 파일은 안 쓴다")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'문서':10s} {'1단':>18s} {'2단':>18s}   채택")
    print("-" * 62)

    for pdf in args.pdfs:
        doc_id = args.doc_id if (args.doc_id and len(args.pdfs) == 1) else pdf.stem
        best = None
        cells = []
        for label, kwargs in (("1단", {}), ("2단", {"two_column": True})):
            try:
                text, dropped, markers = pdf_to_text(pdf, **kwargs)
            except Exception as e:                      # 한쪽만 실패할 수 있다
                cells.append(f"{'실패':>18s}")
                continue
            pct, n, avg = quality(text)
            cells.append(f"{pct * 100:6.1f}% ({n:3d}문장)")
            if best is None or pct > best[0]:
                best = (pct, label, text, dropped, markers)

        if best is None:
            print(f"{doc_id:10s} {'양쪽 다 실패':>18s}")
            continue
        pct, label, text, dropped, markers = best
        print(f"{doc_id:10s} {cells[0]:>18s} {cells[1] if len(cells) > 1 else '':>18s}   {label}")

        if not args.dry_run:
            (args.out_dir / f"{doc_id}.txt").write_text(text, encoding="utf-8")
            (args.out_dir / f"{doc_id}.review.txt").write_text(
                f"# 추출 방식: {label} (종결어미 {pct * 100:.1f}%)\n" + "\n".join(dropped),
                encoding="utf-8")
    if not args.dry_run:
        print(f"\n-> {args.out_dir}  (초벌이다. review 파일을 보고 손으로 검수할 것)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
