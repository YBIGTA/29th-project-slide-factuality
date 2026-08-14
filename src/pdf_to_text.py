"""PDF -> 초벌 텍스트 (정제 보조 도구)

A-2 본체가 아니다. `docs/clean/` 을 손으로 만들 때 출발점을 주는 도구다.
**출력물은 반드시 사람이 검수해야 한다.** 같이 나오는 `.review.txt` 를 보고
무엇이 지워졌는지 확인한 뒤 `docs/clean/{doc_id}.txt` 로 옮긴다.

하는 일
  1) pypdf layout 모드로 추출        (기본 모드는 한글 공백이 전부 사라진다)
  2) 머리말/꼬리말 제거              (`132 한국예술연구 제52호` 같은 줄)
  3) 각주 블록 제거                  (페이지 하단의 `8) Jonathan Glasser, ...`)
  4) 줄바꿈 복원                     (`작동했`+`다.` 는 붙이고 `문법을`+`통해` 는 띄운다)
  5) 본문 속 각주 마커 제거          (`지적한다.8)` -> `지적한다.`)

사용법
  python src/pdf_to_text.py "논문.pdf" --doc-id arts_01
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

try:
    from kiwipiepy import Kiwi
except ImportError:  # 줄 이음매 판단만 못 하게 된다
    Kiwi = None


# ─────────────────────────────────────────────────────────────
# 줄 이음매 판단
# ─────────────────────────────────────────────────────────────

# 이 태그로 끝나면 어절이 완결된 것이므로 다음 줄과 띄운다
_TAIL_SPACE = ("JK", "JX", "JC", "EF", "EC", "ETN", "ETM",
               "SF", "SP", "SE", "SS", "SW", "SO")
# 이 태그로 끝나면 뒤에 뭐가 더 붙어야 하므로 다음 줄과 붙인다
#   VV/VA : 동사·형용사 어간은 어미 없이 어절을 끝낼 수 없다 ("메" + "웠으며")
#   EP    : 선어말어미도 마찬가지 ("작동했" + "다.")
_TAIL_JOIN = ("VV", "VA", "VX", "VCP", "VCN", "XSV", "XSA", "EP", "XPN")


class LineJoiner:
    """줄 끝과 다음 줄 시작을 붙일지 띄울지 판단한다."""

    def __init__(self, kiwi=None, ctx: int = 14):
        self.kiwi = kiwi
        self.ctx = ctx

    def needs_space(self, left: str, right: str) -> bool:
        if not left or not right:
            return False
        # 한글이 아닌 쪽이 끼면 그냥 띄운다 (라틴 문헌명 등)
        if not (_is_hangul(left[-1]) and _is_hangul(right[0])):
            return True
        if self.kiwi is None:
            return True  # kiwi 없으면 안전하게 띄운다 (붙이면 없는 단어가 생긴다)

        tail, head = left[-self.ctx:], right[:self.ctx]

        # ① 오른쪽이 어미·조사로 시작하면 어절 중간이 잘린 것이다 -> 붙인다
        #    ("보여준" + "다." / "메" + "웠으며" / "음악" + "을 통해")
        #    왼쪽만 보면 "보여준"이 관형형으로도 읽혀 판단이 갈린다. 오른쪽이 더 확실하다.
        rtoks = [t for t in self.kiwi.tokenize(head) if t.len > 0]
        if rtoks and rtoks[0].tag.startswith(("E", "J")):
            return False

        # ② 왼쪽이 조사·어미·문장부호로 끝나면 어절이 완결된 것이다 -> 띄운다
        ltoks = [t for t in self.kiwi.tokenize(tail) if t.len > 0]
        if ltoks:
            tag = ltoks[-1].tag
            if tag.startswith(_TAIL_SPACE):
                return True
            if tag.startswith(_TAIL_JOIN):
                return False

        # ③ 체언으로 끝났으면 복합명사인지 두 단어인지 알 수 없다 -> 분석 점수로 결정
        return self._score(tail + " " + head) > self._score(tail + head)

    def _score(self, s: str) -> float:
        res = self.kiwi.analyze(s, top_n=1)
        return res[0][1] if res else -1e9


def _is_hangul(ch: str) -> bool:
    return "가" <= ch <= "힣"


# ─────────────────────────────────────────────────────────────
# 페이지 구성요소 제거
# ─────────────────────────────────────────────────────────────

_RUNNING_HEAD = re.compile(r"^\d{1,4}\s+\S|\s\d{1,4}$")
_FOOTNOTE_HEAD = re.compile(r"^(\d{1,2})\)")


def is_running_head(line: str) -> bool:
    """`132 한국예술연구 제52호` / `임성희 유대-아랍 음악과 문화정체성 135`"""
    s = line.strip()
    if not s or len(s) > 45:
        return False
    if s.endswith((".", "다", "?", "!", ":")):
        return False
    return bool(_RUNNING_HEAD.search(s))


def _latin_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c.isascii() for c in letters) / len(letters)


def split_footnotes(lines: list[str], marker_nums: set[int], body_floor: float = 0.4):
    """페이지 하단의 각주 블록을 잘라낸다.

    각주는 페이지 아래쪽에 몰려 있고 `8)` 처럼 번호로 시작한다.
    앞 페이지에서 넘어온 각주 이어짐 줄(번호 없는 라틴 문헌명)도 같이 딸려간다.

    함정: 절 제목도 `1) 간전기의 역설:` 처럼 번호로 시작하고 페이지 하단에 올 수 있다.
    번호만 보고 자르면 **본문 문단이 통째로 사라진다.**
    그래서 `marker_nums` — 본문에 실제로 달려 있던 각주 마커 번호 — 에 있는 번호만
    각주로 인정한다. 절 제목의 `1)` `2)` 는 본문 마커가 아니므로 살아남는다.
    """
    n = len(lines)
    if n < 4:
        return lines, []
    floor = int(n * body_floor)
    start = None
    for i in range(floor, n):
        m = _FOOTNOTE_HEAD.match(lines[i])
        is_note = (m and int(m.group(1)) in marker_nums) or _latin_ratio(lines[i]) > 0.85
        if is_note:
            start = i
            break
    if start is None:
        return lines, []
    return lines[:start], lines[start:]


# ─────────────────────────────────────────────────────────────
# 본문 정리
# ─────────────────────────────────────────────────────────────

# 본문에 남은 각주 마커: `지적한다.8)` `묘사11)했으나` `외부자’16)로`
_INLINE_MARKER = re.compile(r"(?<=[가-힣.’”\"'\)])(\d{1,2})\)")

_CHAR_FIX = {
    "․": "·",   # ․ ONE DOT LEADER -> 가운뎃점 (마침표로 오인된다)
    "∙": "·",
    "．": ".",
    " ": " ",
}


def clean_body(text: str) -> str:
    for bad, good in _CHAR_FIX.items():
        text = text.replace(bad, good)
    text = _INLINE_MARKER.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────

# 공백이 통째로 빠진 구간. pypdf가 어떤 줄에서는 한글 공백을 못 살린다.
_LONG_RUN = re.compile(r"[가-힣]{8,}")


def fix_long_runs(text: str, kiwi, log: list[str]) -> str:
    """공백 없이 붙어버린 긴 한글 덩어리에만 kiwi 띄어쓰기를 적용한다.

    kiwi.space() 를 문서 전체에 돌리면 `안달루시아`->`안달 루시아`,
    `20세기`->`20 세기` 처럼 멀쩡한 곳까지 망가진다. 그래서 진짜 붙어 있는
    구간만 골라 고치고, 고친 내역은 전부 로그로 남겨 사람이 확인하게 한다.
    """
    if kiwi is None:
        return text

    def rep(m):
        before = m.group(0)
        after = kiwi.space(before)
        if after != before:
            log.append(f"[띄어쓰기] {before}  ->  {after}")
        return after

    return _LONG_RUN.sub(rep, text)


# 절 제목: `2) 마그레브 유대인과 음악: 공유된 전통과 차별화된 실천`
#          `3. 간전기에서 독립전쟁기: 유대-아랍 음악의 균열과 종언`
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+)[.)]\s*\S")


def is_heading(line: str) -> bool:
    s = line.strip()
    return bool(_HEADING.match(s)) and len(s) <= 60 and not s.endswith(("다.", "다"))


def _page_lines(page) -> list[tuple[int, str]]:
    """(들여쓰기 칸 수, 줄 내용). 들여쓰기는 문단 시작을 알려주는 유일한 단서다."""
    raw = page.extract_text(extraction_mode="layout") or ""
    out = []
    for ln in raw.split("\n"):
        if not ln.strip():
            continue
        indent = len(ln) - len(ln.lstrip(" "))
        out.append((indent, re.sub(r"\s+", " ", ln).strip()))
    return out


def pdf_to_text(pdf_path: str | Path, keep_footnotes: bool = False,
                fix_spacing: bool = True):
    reader = PdfReader(str(pdf_path))
    joiner = LineJoiner(Kiwi() if Kiwi else None)
    pages = [_page_lines(p) for p in reader.pages]

    # 1차 통과: 본문에 실제로 달려 있는 각주 마커 번호를 모은다.
    #           이게 있어야 각주와 절 제목을 구분할 수 있다.
    marker_nums = {
        int(n) for lines in pages
        for n in _INLINE_MARKER.findall(" ".join(ln for _, ln in lines))
    }

    # 문단 단위로 모은다. 문단 = 들여쓰기로 시작하는 줄부터 다음 들여쓰기 직전까지.
    paragraphs: list[list[str]] = []
    dropped: list[str] = []

    for pno, lines in enumerate(pages, 1):
        kept = [(ind, ln) for ind, ln in lines if not is_running_head(ln)]
        for _, ln in lines:
            if is_running_head(ln):
                dropped.append(f"[p{pno} 머리말] {ln}")

        body_txt, notes = split_footnotes([ln for _, ln in kept], marker_nums)
        for ln in notes:
            dropped.append(f"[p{pno} 각주]   {ln}")
        body = kept[:len(body_txt)]
        if keep_footnotes:
            body += [(0, ln) for ln in notes]
        if not body:
            continue

        base = min(ind for ind, _ in body)
        for i, (indent, ln) in enumerate(body):
            head = is_heading(ln)
            # 페이지 첫 줄은 앞 페이지 문단이 이어지는 경우가 대부분이므로 들여쓰기를 믿지 않는다
            first_of_page = (i == 0 and pno > 1)
            start_new = head or (indent > base and not first_of_page) or not paragraphs
            if start_new:
                paragraphs.append([ln])
            else:
                paragraphs[-1].append(ln)
            if head:
                paragraphs.append([])          # 제목 다음은 무조건 새 문단
        paragraphs = [p for p in paragraphs if p]

    # 문단 안에서만 줄바꿈을 복원한다 (문장이 페이지를 넘어가도 문단은 이어진다)
    blocks = []
    for para in paragraphs:
        s = para[0]
        for nxt in para[1:]:
            s += (" " if joiner.needs_space(s, nxt) else "") + nxt
        blocks.append(s)

    out = clean_body("\n\n".join(blocks))
    if fix_spacing:
        out = fix_long_runs(out, joiner.kiwi, dropped)
    return out, dropped, sorted(marker_nums)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--out-dir", default="docs/clean")
    ap.add_argument("--keep-footnotes", action="store_true")
    ap.add_argument("--no-fix-spacing", action="store_true")
    args = ap.parse_args()

    text, dropped, markers = pdf_to_text(
        args.pdf, args.keep_footnotes, fix_spacing=not args.no_fix_spacing)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt = out_dir / f"{args.doc_id}.txt"
    rev = out_dir / f"{args.doc_id}.review.txt"
    txt.write_text(text, encoding="utf-8")
    rev.write_text("\n".join(dropped), encoding="utf-8")

    print(f"본문   {len(text):,}자 -> {txt}")
    print(f"제거   {len(dropped)}줄  -> {rev}")
    print(f"각주   마커 {markers}")
    print("\n!! 초벌이다. review 파일을 보고 본문을 직접 검수한 뒤 라벨링에 쓸 것 !!")


if __name__ == "__main__":
    main()
