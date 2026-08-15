"""A-5 · claim 에 claim_id 를 붙여 검색 쿼리 파일을 만든다

`claims/{deck_id}.jsonl` 에는 **claim_id 가 없다.** 필드가 `Slide #`,
`Claim (PPT)`, `Slide_Title`, `Context (PPT)` 네 개뿐이다. 그래서 BM25 와
임베딩이 각자 ID 를 만들면 서로 결과를 못 맞춘다. 이 스크립트가 ID 를
한 번만 부여하고, 두 검색기는 모두 이 출력을 읽는다.

    claim_id = "{deck_id}_s{슬라이드:02d}_c{슬라이드내순번:02d}"
               예) arts_01__claudecode_s03_c05

파일 전체 줄번호가 아니라 **슬라이드 단위로 끊어서** 센다. 앞 슬라이드에서
claim 이 하나 늘어도 뒷 슬라이드의 ID 가 밀리지 않는다. claims/*.jsonl 은
이미 한 번 통째로 재생성된 이력이 있어서(커밋 a0d8095) 이 성질이 필요하다.

이 스크립트는 torch 없이 돈다. BM25 담당자가 임베딩 스택을 설치하지 않고도
같은 쿼리 파일을 쓸 수 있어야 하기 때문에 일부러 분리했다.

사용법
  python src/build_queries.py --deck-id arts_01__claudecode
  python src/build_queries.py --all
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def load_manifest(root: Path) -> dict[str, str]:
    """deck_id -> doc_id 매핑. manifest 가 정답이다."""
    path = root / "docs" / "manifest.csv"
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            deck_id = (row.get("deck_id") or "").strip()
            doc_id = (row.get("doc_id") or "").strip()
            if deck_id and doc_id:
                mapping[deck_id] = doc_id
    return mapping


def resolve_doc_id(deck_id: str, manifest: dict[str, str]) -> str:
    """manifest 에 없으면 `{doc_id}__{tool}` 규칙으로 추정하고 경고한다."""
    if deck_id in manifest:
        return manifest[deck_id]
    guess = deck_id.split("__")[0]
    print(f"WARN {deck_id}: manifest.csv 에 없다. doc_id 를 '{guess}' 로 추정한다.",
          file=sys.stderr)
    return guess


def normalize_query(text: str) -> str:
    """검색에 넣을 형태. PPT 의 줄바꿈은 조판일 뿐이라 공백으로 편다.

    원문 claim_text 는 그대로 보존한다. 판별 모델과 라벨 시트가 원문을 봐야 한다.
    """
    return re.sub(r"\s+", " ", text).strip()


def build(root: Path, deck_id: str, manifest: dict[str, str]) -> list[dict]:
    path = root / "claims" / f"{deck_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없다")

    doc_id = resolve_doc_id(deck_id, manifest)
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    per_slide: dict[int, int] = {}
    out: list[dict] = []
    for row in rows:
        slide_no = int(row["Slide #"])
        per_slide[slide_no] = per_slide.get(slide_no, 0) + 1
        claim_text = row["Claim (PPT)"]
        out.append({
            "claim_id": f"{deck_id}_s{slide_no:02d}_c{per_slide[slide_no]:02d}",
            "deck_id": deck_id,
            "doc_id": doc_id,
            "slide_no": slide_no,
            "claim_text": claim_text,                    # 원문 그대로
            "query_text": normalize_query(claim_text),   # 검색기가 쓰는 형태
            "slide_title": normalize_query(row.get("Slide_Title", "")),
            "context": row.get("Context (PPT)", []),
        })
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck-id")
    ap.add_argument("--all", action="store_true", help="claims/ 안의 모든 덱")
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()

    if args.all:
        deck_ids = sorted(p.stem for p in (args.root / "claims").glob("*.jsonl"))
    elif args.deck_id:
        deck_ids = [args.deck_id]
    else:
        ap.error("--deck-id 또는 --all 이 필요하다")

    manifest = load_manifest(args.root)
    for deck_id in deck_ids:
        rows = build(args.root, deck_id, manifest)
        out = args.root / "retrieval" / "queries" / f"{deck_id}.jsonl"
        write_jsonl(out, rows)

        pool = args.root / "passages" / f"{rows[0]['doc_id']}.jsonl"
        mark = "" if pool.exists() else "   <- passage pool 없음, 검색 불가"
        slides = len({r["slide_no"] for r in rows})
        short = sum(1 for r in rows if len(r["query_text"]) < 10)
        print(f"OK   {deck_id}: {len(rows)} claims ({slides} slides) -> {out}{mark}")
        print(f"     doc_id={rows[0]['doc_id']}  10자 미만 claim {short}개 "
              f"({short / len(rows):.0%}) — dense 가 약한 구간")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
