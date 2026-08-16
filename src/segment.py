from __future__ import annotations
"""A-2 · 문장 분리기 + 근거 구간 생성

`docs/clean/{doc_id}.txt` 를 받아서
  1) 문장으로 나누고            -> spans/sents/{doc_id}.jsonl
  2) 윈도우 3 / 윈도우 5 구간   -> spans/w3/{doc_id}.jsonl, spans/w5/{doc_id}.jsonl
  3) 라벨 시트의 근거 문장을 구간에 붙이는 find_evidence() 를 제공한다

find_evidence() 가 A-2의 숨은 본체다. 이게 없으면 B-1이 recall 을 계산할 수 없고
라벨의 `evidence_text` 는 그냥 텍스트 조각으로 남는다.

사용법
  python src/segment.py docs/clean/arts_01.txt --doc-id arts_01
  python src/segment.py docs/clean/arts_01.txt --doc-id arts_01 --show 20
"""


import argparse
import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path

from kiwipiepy import Kiwi

_KIWI = None


def kiwi() -> Kiwi:
    global _KIWI
    if _KIWI is None:
        _KIWI = Kiwi()
    return _KIWI


# ─────────────────────────────────────────────────────────────
# 예외 규칙 — 마침표가 문장 끝이 아닌 자리
# ─────────────────────────────────────────────────────────────
# 이 논문에서 실제로 터진 것들을 넣었다. 새 논문에서 깨지면 여기에 추가한다.
_PROTECT = [
    r"\bpp?\.\s*\d",                       # p.679  pp. 682-683
    r"\b(?:Vol|No|Ch|Fig|Tab|Eds?|Trans|Op|cit|cf|[Ii]bid"
    r"|Jr|Sr|St|Mr|Mrs|Ms|Dr|Prof|vs|approx)\.",
    r"\bet\s*al\.",
    r"\b(?:U\.S|U\.K|A\.D|B\.C|e\.g|i\.e|Ph\.D|M\.A|B\.A)\.",
    r"\b[A-Z]\.(?=\s*[A-Z])",              # 이니셜  J. Glasser
    r"\d+\.\d+",                           # 3.5%  0.05
    r"\d{4}\.\s?\d{1,2}\.\s?\d{1,2}\.?",   # 1995.12.6.
    r"\d+\.\d+\.\d+",                      # 절 번호 2.1.3
]
_MASK = "\x00"


def _mask_dots(text: str) -> str:
    """문장 끝이 아닌 마침표를 같은 길이의 다른 글자로 가린다.

    길이를 보존해야 kiwi 가 돌려주는 start/end 오프셋을 원문에 그대로 쓸 수 있다.
    """
    chars = list(text)
    for pat in _PROTECT:
        for m in re.finditer(pat, text):
            for i in range(m.start(), m.end()):
                if chars[i] == ".":
                    chars[i] = _MASK
    return "".join(chars)


# 절 제목: `2) 마그레브 유대인과 음악: ...`  `3. 간전기에서 독립전쟁기: ...`
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+|[가-힣])[.)]\s*\S")

_OPEN_CLOSE = [("(", ")"), ("‘", "’"), ("“", "”"), ("〈", "〉"), ("<", ">"),
               ("《", "》"), ("「", "」"), ("『", "』"), ("[", "]")]

_SENT_END = ("다.", "다", ".", "!", "?", "요.", "함.", "임.", "음.")


def _unbalanced(s: str) -> bool:
    """따옴표·괄호가 안 닫혔으면 문장이 중간에 잘린 것이다."""
    return any(s.count(o) != s.count(c) for o, c in _OPEN_CLOSE)


def is_heading(s: str) -> bool:
    s = s.strip()
    return bool(_HEADING.match(s)) and len(s) <= 60 and not s.endswith("다.")


@dataclass
class Sentence:
    sent_id: str
    text: str
    start: int
    end: int
    is_heading: bool


def split_sentences(text: str, min_len: int = 6) -> list[Sentence]:
    """문단을 유지한 채 문장으로 나눈다.

    문단(빈 줄로 구분)을 넘어가며 문장을 합치지 않는다. 정제 단계에서 살려둔
    문단 경계는 사람이 만든 것이라 kiwi 의 추측보다 정확하다.
    """
    k = kiwi()
    out: list[tuple[str, int, int, bool]] = []

    pos = 0
    for para in text.split("\n\n"):
        block = para.strip()
        if not block:
            pos += len(para) + 2
            continue
        base = text.index(block, pos)

        if is_heading(block):
            out.append((block, base, base + len(block), True))
        else:
            for s in k.split_into_sents(_mask_dots(block)):
                raw = text[base + s.start: base + s.end]
                out.append((raw, base + s.start, base + s.end, False))
        pos = base + len(block)

    # ── 후처리: 잘못 잘린 조각을 앞 문장에 되붙인다
    merged: list[tuple[str, int, int, bool]] = []
    for item in out:
        txt, st, en, head = item
        if merged and not head and not merged[-1][3]:
            ptxt, pst, _, phead = merged[-1]
            too_short = len(txt.strip()) < min_len
            prev_open = _unbalanced(ptxt)
            prev_unfinished = not ptxt.rstrip().endswith(_SENT_END)
            if too_short or prev_open or prev_unfinished:
                merged[-1] = (text[pst:en], pst, en, phead)
                continue
        merged.append(item)

    return [Sentence(f"s{i:03d}", t.strip(), st, en, h)
            for i, (t, st, en, h) in enumerate(merged, 1)]


# ─────────────────────────────────────────────────────────────
# 근거 구간 (윈도우)
# ─────────────────────────────────────────────────────────────

@dataclass
class Span:
    span_id: str
    text: str
    sent_ids: list[str]


def build_spans(sents: list[Sentence], window: int = 3, stride: int = 1) -> list[Span]:
    if not sents:
        return []
    if len(sents) <= window:
        return [Span(f"w{window}_001", " ".join(s.text for s in sents),
                     [s.sent_id for s in sents])]
    spans = []
    for i in range(0, len(sents) - window + 1, stride):
        chunk = sents[i:i + window]
        spans.append(Span(f"w{window}_{len(spans) + 1:03d}",
                          " ".join(s.text for s in chunk),
                          [s.sent_id for s in chunk]))
    return spans


# ─────────────────────────────────────────────────────────────
# 근거 문장 -> 구간 정렬  (B-1 의 recall 계산, B-2 의 dataset 조립에 쓰인다)
# ─────────────────────────────────────────────────────────────

_QUOTES = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"', "＇": "'", "＂": '"',
    "․": "·", "∙": "·", "·": "·", "―": "-", "–": "-", "—": "-",
})


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_QUOTES)
    s = re.sub(r"(?<=[가-힣.'\")])\d{1,2}\)", "", s)   # 남아 있는 각주 마커
    return re.sub(r"\s+", " ", s).strip()


def _nospace(s: str) -> str:
    return re.sub(r"\s+", "", normalize(s))


@dataclass
class EvidenceMatch:
    sent_ids: list[str]
    span_ids: list[str]
    method: str          # exact | nospace | fuzzy | none
    score: float


def find_evidence(sents: list[Sentence], spans: list[Span], evidence: str,
                  fuzzy_floor: float = 0.85, max_join: int = 3) -> EvidenceMatch:
    """라벨 시트의 `evidence_text` 가 어느 문장·구간에 있는지 찾는다.

    복붙이 완벽하지 않을 것을 전제한다. 세 단계로 느슨해진다.
      exact   정규화 후 그대로 포함
      nospace 공백을 전부 지우고 포함        (띄어쓰기만 틀린 경우)
      fuzzy   유사도 fuzzy_floor 이상         (조사 한두 개가 다른 경우)
    셋 다 실패하면 method='none' 이다. **조용히 버리지 말고 사람이 고쳐야 한다.**
    """
    if not evidence or not evidence.strip():
        return EvidenceMatch([], [], "none", 0.0)

    # 가이드에서 근거 2개는 ' || ' 로 나눠 적게 했다
    parts = [p for p in re.split(r"\s*\|\|\s*", evidence) if p.strip()]
    if len(parts) > 1:
        hits, best, method = [], 0.0, "exact"
        for p in parts:
            m = find_evidence(sents, spans, p, fuzzy_floor, max_join)
            hits += m.sent_ids
            best = max(best, m.score)
            if m.method == "none":
                method = "none"
            elif method != "none" and m.method != "exact":
                method = m.method
        uniq = list(dict.fromkeys(hits))
        return EvidenceMatch(uniq, _spans_for(spans, uniq), method, best)

    ev, evn = normalize(evidence), _nospace(evidence)

    # 1~max_join 개 연속 문장을 이어붙여 가며 찾는다 (근거가 두 문장에 걸칠 수 있다)
    for n in range(1, max_join + 1):
        for i in range(len(sents) - n + 1):
            group = sents[i:i + n]
            joined = " ".join(s.text for s in group)
            if ev in normalize(joined):
                ids = [s.sent_id for s in group]
                return EvidenceMatch(ids, _spans_for(spans, ids), "exact", 1.0)
    for n in range(1, max_join + 1):
        for i in range(len(sents) - n + 1):
            group = sents[i:i + n]
            joined = " ".join(s.text for s in group)
            if evn and evn in _nospace(joined):
                ids = [s.sent_id for s in group]
                return EvidenceMatch(ids, _spans_for(spans, ids), "nospace", 1.0)

    best_ids, best_score = [], 0.0
    for n in range(1, max_join + 1):
        for i in range(len(sents) - n + 1):
            group = sents[i:i + n]
            cand = _nospace(" ".join(s.text for s in group))
            r = SequenceMatcher(None, evn, cand).ratio()
            if r > best_score:
                best_score, best_ids = r, [s.sent_id for s in group]
    if best_score >= fuzzy_floor:
        return EvidenceMatch(best_ids, _spans_for(spans, best_ids), "fuzzy", best_score)
    return EvidenceMatch([], [], "none", best_score)


def _spans_for(spans: list[Span], sent_ids: list[str]) -> list[str]:
    """근거 문장을 **전부** 담은 구간. 없으면 하나라도 겹치는 구간.

    윈도우가 겹치므로 보통 여러 개가 나온다. 이 집합이 검색 평가의 정답이다.
    top-k 안에 이 중 하나라도 있으면 hit 으로 센다.
    """
    if not sent_ids:
        return []
    want = set(sent_ids)
    full = [sp.span_id for sp in spans if want <= set(sp.sent_ids)]
    if full:
        return full
    return [sp.span_id for sp in spans if want & set(sp.sent_ids)]


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("txt")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--show", type=int, default=0, help="문장 N개를 찍어본다")
    args = ap.parse_args()

    text = Path(args.txt).read_text(encoding="utf-8")
    sents = split_sentences(text)
    w3, w5 = build_spans(sents, 3), build_spans(sents, 5)

    root = Path(args.out)
    _write_jsonl(root / "spans" / "sents" / f"{args.doc_id}.jsonl", sents)
    _write_jsonl(root / "spans" / "w3" / f"{args.doc_id}.jsonl", w3)
    _write_jsonl(root / "spans" / "w5" / f"{args.doc_id}.jsonl", w5)

    body = [s for s in sents if not s.is_heading]
    lens = sorted(len(s.text) for s in body)
    print(f"문장 {len(sents)}개 (제목 {len(sents) - len(body)})  "
          f"w3 {len(w3)}구간  w5 {len(w5)}구간")
    if body:
        print(f"문장 길이  중앙값 {lens[len(lens) // 2]}자  "
              f"최소 {lens[0]}  최대 {lens[-1]}")
    long_ones = [s for s in body if len(s.text) > 250]
    if long_ones:
        print(f"\n[확인 필요] 250자 넘는 문장 {len(long_ones)}개 — 분리 실패일 수 있다")
        for s in long_ones[:3]:
            print(f"  {s.sent_id} ({len(s.text)}자) {s.text[:60]}...")

    for s in sents[:args.show]:
        tag = "제목" if s.is_heading else "  "
        print(f"{s.sent_id} {tag} {s.text}")


if __name__ == "__main__":
    main()
