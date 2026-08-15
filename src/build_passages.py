"""A-5 · passage pool 생성 — 검색기(BM25 · 임베딩)의 공용 입력을 만든다

`spans/{window}/{doc_id}.jsonl` 을 받아서 `passages/{doc_id}.jsonl` 로 옮긴다.
하는 일은 사실상 하나다. **문서 간 고유한 passage_id 를 붙이는 것.**

A-2 가 만든 span_id 는 문서 안에서만 유일하다. `w3_001` 은 모든 문서에 있다.
검색 결과를 문서 간에 섞거나 BM25 결과와 융합하려면 전역 ID 가 필요하다.

    passage_id = "{doc_id}_{span_id}"        예) arts_01_w3_007

접두사만 붙이는 이유는 되돌릴 수 있어야 하기 때문이다. B-1 의 정답 집합은
`segment.find_evidence(...).span_ids` 가 돌려주는 **원래 span_id** 로 되어 있다.
passage_id 에서 doc_id 를 떼면 그대로 정답과 맞출 수 있다.

`{doc_id}_p7` 같은 형태를 쓰지 않은 것도 같은 이유다. `p7` 은 어떤 청킹으로
만든 7번인지를 잃어버린다. `w3_007` 은 청킹 단위가 ID 안에 남아 있어서
w3 pool 과 w5 pool 을 한 파일에 섞어도 충돌하지 않는다.

사용법
  python src/build_passages.py --doc-id arts_01
  python src/build_passages.py --doc-id arts_01 --window w5
  python src/build_passages.py --all                    # spans/w3 안의 모든 문서
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_WINDOW = "w3"


def parse_passage_id(passage_id: str) -> tuple[str, str]:
    """passage_id 를 (doc_id, span_id) 로 되돌린다.

    span_id 는 항상 `w{숫자}_{숫자}` 형태이므로 뒤에서 두 조각이 span_id 다.
    doc_id 에 밑줄이 들어 있어도(`arts_01`) 안전하다.

    >>> parse_passage_id("arts_01_w3_007")
    ('arts_01', 'w3_007')
    """
    head, _, tail = passage_id.rpartition("_")       # head='arts_01_w3', tail='007'
    doc_id, _, window = head.rpartition("_")         # doc_id='arts_01', window='w3'
    return doc_id, f"{window}_{tail}"


def load_spans(root: Path, doc_id: str, window: str) -> list[dict]:
    path = root / "spans" / window / f"{doc_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없다. 먼저 A-2 를 돌려야 한다:\n"
            f"  python src/pdf_to_text.py docs/raw/{doc_id}.pdf --doc-id {doc_id}\n"
            f"  python src/segment.py docs/clean/{doc_id}.txt --doc-id {doc_id}"
        )
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build(root: Path, doc_id: str, window: str = DEFAULT_WINDOW) -> list[dict]:
    spans = load_spans(root, doc_id, window)
    passages = []
    for sp in spans:
        span_id = sp["span_id"]
        passages.append({
            "passage_id": f"{doc_id}_{span_id}",
            "doc_id": doc_id,
            "span_id": span_id,          # B-1 정답 집합과 맞추는 키
            "window": window,
            "text": sp["text"],
            "sent_ids": sp["sent_ids"],  # 겹치는 passage 를 뒤에서 합칠 때 쓴다
            "n_chars": len(sp["text"]),
        })
    return passages


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc-id")
    ap.add_argument("--all", action="store_true", help="spans/{window} 안의 모든 문서")
    ap.add_argument("--window", default=DEFAULT_WINDOW, choices=["w3", "w5"])
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()

    if args.all:
        doc_ids = sorted(p.stem for p in (args.root / "spans" / args.window).glob("*.jsonl"))
    elif args.doc_id:
        doc_ids = [args.doc_id]
    else:
        ap.error("--doc-id 또는 --all 이 필요하다")

    if not doc_ids:
        print("처리할 문서가 없다.")
        return 2

    for doc_id in doc_ids:
        rows = build(args.root, doc_id, args.window)
        # w3 는 기본 pool 이라 파일명에 window 를 넣지 않는다. w5 는 구분한다.
        name = f"{doc_id}.jsonl" if args.window == DEFAULT_WINDOW else f"{doc_id}.{args.window}.jsonl"
        out = args.root / "passages" / name
        write_jsonl(out, rows)
        lens = sorted(r["n_chars"] for r in rows)
        print(f"OK   {doc_id}: {len(rows)} passages -> {out}")
        print(f"     길이 중앙값 {lens[len(lens) // 2]}자  최소 {lens[0]}  최대 {lens[-1]}")
        print(f"     예시 passage_id: {rows[0]['passage_id']} ... {rows[-1]['passage_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
