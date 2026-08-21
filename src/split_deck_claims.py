#!/usr/bin/env python3
"""덱 claim 중 여러 문장이 한 칸에 묶인 것을 문장 단위로 쪼갠다.

claim_split.py 는 슬라이드의 '본문 상자 하나'를 claim 하나로 뽑는다.
그 안에 불릿이 세 줄이면 세 주장이 한 claim 에 뭉쳐 있어서 라벨을
하나만 붙일 수 없다 — 한 줄은 근거가 있고 다른 줄은 무근거일 수 있다.

줄바꿈으로 쪼개되, **20자 이상인 조각이 둘 이상일 때만** 쪼갠다.
제목이 두 줄로 접힌 것("딥러닝 모델을 이용한 / 고주파 연소 불안정 조기 감지")은
둘 다 짧아서 안 쪼개지고, 머리말 한 줄에 긴 불릿이 여러 개 달린 것
("주요 포인트 / • 난민에 대한 수용성이 ... / • 사회 신뢰가 ...")은 쪼개진다.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MIN_PART = 20


def parts(text: str) -> list[str]:
    chunks = [c.strip() for c in text.split("\n") if c.strip()]
    if sum(len(c) >= MIN_PART for c in chunks) < 2:
        return [text.strip()]
    return chunks


def build(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        key = "Claim (PPT)" if "Claim (PPT)" in r else "claim_text"
        slide = r.get("Slide #") or r.get("slide")
        for p in parts(str(r[key])):
            out.append({"slide": slide, "claim": p, "tag": r.get("tag", "")})
    # 슬라이드별 번호를 다시 매긴다
    per: dict = {}
    for row in out:
        per[row["slide"]] = per.get(row["slide"], 0) + 1
        row["claim_id"] = f"s{int(row['slide']):02d}_c{per[row['slide']]:02d}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("claims", type=Path)
    args = ap.parse_args()
    rows = build(args.claims)
    for r in rows:
        print(f"{r['claim_id']} | {r['claim']}")
    print(f"\n{args.claims.stem}: {rows[-1] and len(rows)} claim")


if __name__ == "__main__":
    main()
