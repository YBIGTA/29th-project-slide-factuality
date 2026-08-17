"""A-2 테스트 — 예외 규칙과 근거 정렬

  python src/test_segment.py

새 논문을 정제하다가 이상하게 잘리는 문장을 발견하면
여기에 케이스를 추가하고 segment.py 의 _PROTECT 를 고친다.
"""
from pathlib import Path

from segment import split_sentences, build_spans, find_evidence, normalize

OK, NG = "  OK  ", "  !! FAIL"
fails = 0


def check(name, got, want):
    global fails
    good = got == want
    if not good:
        fails += 1
    print(f"{OK if good else NG}  {name}")
    if not good:
        print(f"        기대 {want!r}")
        print(f"        실제 {got!r}")


# ─────────────────────────────────────────────────────────────
print("\n[1] 마침표 예외 — 여기서 잘리면 안 된다")
# ─────────────────────────────────────────────────────────────
cases = [
    ("서지 쪽수",   "이는 Glasser, Op. cit., 2012, pp.682-683 에 정리되어 있다. 다음 절에서 다룬다.", 2),
    ("권호 표기",   "해당 논문은 Vol.44, No.4 에 실렸다.", 1),
    ("et al.",      "Silver et al. 은 이를 반박했다.", 1),
    ("소수점",      "판매량은 전년 대비 4.3배 증가했다.", 1),
    ("날짜",        "이 기사는 1995.12.6. 자 El Watan 에 실렸다.", 1),
    ("이니셜",      "이는 J. Glasser 의 연구에서 확인된다.", 1),
    ("두 문장",     "협회가 창설되었다. 이후 주류에서 밀려났다.", 2),
    ("붙은 두 문장", "협회가 창설되었다.이후 주류에서 밀려났다.", 2),
]
for name, text, want in cases:
    check(f"{name:10s} {text[:34]}", len(split_sentences(text)), want)

# ─────────────────────────────────────────────────────────────
print("\n[2] 절 제목은 본문과 섞이지 않는다")
# ─────────────────────────────────────────────────────────────
doc = "3) 식민주의와 위계의 전도: 음반 산업의 선점\n\n1870년의 크레미외 법령은 시민권을 부여했다. 이는 전환점이었다."
s = split_sentences(doc)
check("제목 1 + 본문 2", [x.is_heading for x in s], [True, False, False])

doc = "표 제목\n\n첫 문장은 문단 경계를 넘어 앞 조각에 붙으면 안 된다."
s = split_sentences(doc)
check("빈 줄 문단 경계 유지", [x.text for x in s],
      ["표 제목", "첫 문장은 문단 경계를 넘어 앞 조각에 붙으면 안 된다."])

# ─────────────────────────────────────────────────────────────
print("\n[3] 인용부호 안에서 잘리면 되붙인다")
# ─────────────────────────────────────────────────────────────
q = "그들은 ‘음악적 순수성’을 무기로 삼았다."
check("따옴표 유지", len(split_sentences(q)), 1)

# ─────────────────────────────────────────────────────────────
print("\n[4] 근거 정렬 — 실제 논문으로")
# ─────────────────────────────────────────────────────────────
txt = Path("docs/clean/arts_01.txt").read_text(encoding="utf-8")
sents = split_sentences(txt)
w3 = build_spans(sents, 3)
w5 = build_spans(sents, 5)
print(f"        문장 {len(sents)}  w3 {len(w3)}  w5 {len(w5)}")

target = next(s for s in sents if "20만 장" in s.text)
print(f"        기준 문장 {target.sent_id}: {target.text[:56]}...")

trials = [
    ("원문 그대로 복붙",   target.text,                              "exact"),
    ("띄어쓰기 틀림",      target.text.replace(" ", ""),             "nospace"),
    ("일부만 복붙",        target.text[10:70],                       "exact"),
    ("따옴표 종류 다름",   target.text.replace("‘", "'").replace("’", "'"), "exact"),
    ("각주 마커 딸려옴",   target.text + "14)",                      "exact"),
    ("조사 하나 다름",     target.text.replace("이들의", "이 들의"),  "nospace"),
    ("원문에 없는 문장",   "유대인 음악가들은 1960년에 이스라엘로 이주했다.", "none"),
]
for name, ev, want in trials:
    m = find_evidence(sents, w3, ev)
    hit = target.sent_id in m.sent_ids
    ok = m.method == want and (hit or want == "none")
    print(f"{OK if ok else NG}  {name:16s} method={m.method:8s} "
          f"score={m.score:.2f} sents={m.sent_ids} spans={len(m.span_ids)}")
    if not ok:
        fails += 1

# 근거 2개를 || 로 적은 경우
two = sents[20].text + " || " + sents[40].text
m = find_evidence(sents, w3, two)
check("|| 로 근거 2개", sorted(m.sent_ids), sorted([sents[20].sent_id, sents[40].sent_id]))

# ─────────────────────────────────────────────────────────────
print("\n[5] 정답 구간 집합 — 윈도우가 겹치므로 여러 개가 나와야 정상")
# ─────────────────────────────────────────────────────────────
m3 = find_evidence(sents, w3, target.text)
m5 = find_evidence(sents, w5, target.text)
print(f"        w3 정답 구간 {m3.span_ids}")
print(f"        w5 정답 구간 {m5.span_ids}")
check("w3 정답 3개", len(m3.span_ids), 3)
check("w5 정답 5개", len(m5.span_ids), 5)

# 모든 문장이 최소 한 구간에는 들어가야 한다 (검색 대상에서 새는 문장이 없어야 함)
covered = {sid for sp in w3 for sid in sp.sent_ids}
check("w3 가 전 문장 포함", len(covered), len(sents))

print(f"\n{'전부 통과' if not fails else f'{fails}건 실패'}")
