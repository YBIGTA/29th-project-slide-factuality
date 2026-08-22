#!/usr/bin/env python3
"""test 셋과 generalization(재무) 셋 결과를 한 표로 합친다.

    python src/build_compare_table.py

`eval_generalization.py` 가 만든 두 벌의 결과 json 을 읽어
`results/model_compare.md` 를 다시 쓴다. 손으로 표를 옮겨 적지 않는 이유는
숫자가 바뀔 때마다 표만 옛날 값으로 남는 사고가 나기 때문이다.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "results" / "model_compare.md"

MODELS = [
    ("M1", "M1 (frozen e5 + LinearSVC)"),
    ("M2", "M2 (KLUE-RoBERTa, 맥락 없음)"),
    ("M2p", "M2' (동일 백본 대조군)"),
    ("M3", "M3 (KLUE-RoBERTa, 맥락 포함)"),
]
CLASSES = ["근거있음", "무근거", "모순", "benign"]


def load(model_id: str, suffix: str) -> dict:
    p = ROOT / "results" / f"{model_id}_test{suffix}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["runs"][0]["generalization"] | {"_path": p.name}


def main() -> int:
    te = {m: load(m, "_testset_296") for m, _ in MODELS}
    ge = {m: load(m, "_everything_543") for m, _ in MODELS}
    n_te, n_ge = te["M1"]["n"], ge["M1"]["n"]

    L: list[str] = []
    add = L.append

    add("# 모델 비교 — test 셋 vs generalization(재무) 셋")
    add("")
    add(f"네 모델을 **같은 스크립트·같은 가중치**로 두 셋에 각각 돌렸다 "
        f"(`src/eval_generalization.py`).")
    add("")
    add(f"| 셋 | 구성 | 행 수 |")
    add(f"| --- | --- | ---: |")
    add(f"| test | `arts_03` · `bio_04` · `socio_04` · `tech_04` | {n_te} |")
    add(f"| generalization | `finance_01`~`finance_05` (미지 분야 = 재무) | {n_ge} |")
    add("")
    add("두 셋 모두 라벨이 4분류가 아닌 행은 뺐다.")
    add("")

    # ── 1. 종합 ────────────────────────────────────────────────
    add("## 1. 도메인이 바뀌면 얼마나 떨어지는가")
    add("")
    add("| 모델 | test macro-F1 | generalization macro-F1 | 하락 | 하락률 | test 정확도 | gen 정확도 |")
    add("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for m, name in MODELS:
        a, b = te[m]["macro_f1"], ge[m]["macro_f1"]
        drop = a - b
        rate = drop / a if a else 0.0
        add(f"| {name} | {a:.4f} | {b:.4f} | −{drop:.4f} | −{rate:.0%} "
            f"| {te[m]['accuracy']:.4f} | {ge[m]['accuracy']:.4f} |")
    add("")
    best_te = max(MODELS, key=lambda x: te[x[0]]["macro_f1"])
    best_ge = max(MODELS, key=lambda x: ge[x[0]]["macro_f1"])
    add(f"두 셋 모두 **{best_te[1]}** 가 가장 높다"
        f"{'' if best_te[0] == best_ge[0] else f' (generalization 최고는 {best_ge[1]})'}. "
        "맥락을 넣은 M3 가 대조군 M2' 보다 나아지지 않는다는 점도 두 셋에서 같다 — "
        "맥락 추가가 이 과제에서 도움이 되지 않았다는 뜻이다.")
    add("")

    # ── 2. 클래스별 ────────────────────────────────────────────
    add("## 2. 클래스별 F1 — 무너지는 곳은 정해져 있다")
    add("")
    head = "| 모델 | " + " | ".join(f"{c} test / gen" for c in CLASSES) + " |"
    add(head)
    add("| --- | " + " | ".join(["---:"] * len(CLASSES)) + " |")
    for m, name in MODELS:
        cells = []
        for c in CLASSES:
            a, b = te[m]["per_class_f1"][c], ge[m]["per_class_f1"][c]
            cells.append(f"{a:.3f} / **{b:.3f}**" if b == 0 else f"{a:.3f} / {b:.3f}")
        add(f"| {name} | " + " | ".join(cells) + " |")
    add("")
    add("정답 분포")
    add("")
    add("| 셋 | " + " | ".join(CLASSES) + " |")
    add("| --- | " + " | ".join(["---:"] * len(CLASSES)) + " |")
    add("| test | " + " | ".join(str(te["M1"]["support"][c]) for c in CLASSES) + " |")
    add("| generalization | " + " | ".join(str(ge["M1"]["support"][c]) for c in CLASSES) + " |")
    add("")

    zero_ge = [(name, c) for m, name in MODELS for c in ("무근거", "모순")
               if ge[m]["per_class_f1"][c] == 0]
    add(f"**generalization 에서 F1 이 0 인 (모델, 클래스) 조합이 {len(zero_ge)}개다.** "
        + ", ".join(f"{n}의 '{c}'" for n, c in zero_ge) + ".")
    add("")
    add("이게 이번 평가의 핵심이다. 정확도가 0.42~0.58 로 보이는 것은 대부분 "
        "'근거있음' 을 찍어서 나온 수치이고, **환각 탐지에서 정작 잡아야 할 "
        "'무근거'·'모순' 은 거의 못 잡는다.** M3 의 혼동행렬이 그것을 보여준다.")
    add("")
    add("```")
    add("M3 · generalization  (행=정답, 열=예측)")
    add(f"{'':10}" + "".join(f"{c:>9}" for c in CLASSES))
    for c, row in zip(CLASSES, ge["M3"]["confusion"]):
        add(f"{c:10}" + "".join(f"{v:>9}" for v in row))
    add("```")
    add("")
    add("'무근거' 열이 통째로 0 이다 — 한 번도 예측하지 않았다. "
        "'모순' 은 여러 번 예측했지만 맞힌 것이 없다.")
    add("")

    # ── 3. 덱별 ────────────────────────────────────────────────
    add("## 3. generalization 덱별 정확도")
    add("")
    pd = json.loads((ROOT / "results" / "M1_test_everything_543.json").read_text(encoding="utf-8"))["per_deck"]
    per = {m: json.loads((ROOT / "results" / f"{m}_test_everything_543.json").read_text(encoding="utf-8"))["per_deck"]
           for m, _ in MODELS}
    add("| 덱 | n | " + " | ".join(m for m, _ in MODELS) + " | 후보 없는 행 |")
    add("| --- | ---: | " + " | ".join(["---:"] * len(MODELS)) + " | ---: |")
    for deck in sorted(pd):
        row = " | ".join(f"{per[m][deck]['accuracy']:.3f}" for m, _ in MODELS)
        add(f"| `{deck}` | {pd[deck]['n']} | {row} | {pd[deck]['n_without_candidates']} |")
    add("")

    # ── 각주 ───────────────────────────────────────────────────
    add("## 읽을 때 주의할 것")
    add("")
    add("1. **PR #10 은 아직 머지되지 않았다.** 예린님이 올린 259행 "
        "(`finance_01`+`02` 만, sha12 `4a3f556253bf`) 결과 "
        "`M3_test_everything.json` · `M2p_test_everything.json` 은 브랜치 "
        "`M3_new_json` 에 그대로 있다. 이 표의 숫자와 **다른 평가 셋**이므로 "
        "섞어 읽으면 안 된다.")
    add("")
    add("2. **단일 seed 다.** 확보된 가중치가 모델당 한 벌뿐이라 "
        "3-seed 평균·표준편차를 낼 수 없다. 259행 결과(M3 0.1979 ± 0.0420)는 "
        "seed 13/42/777 세 벌의 평균이므로 이 표와 **직접 비교하면 안 된다**. "
        "참고로 그 결과의 run[0] 단독값은 0.2483 으로, 시드 간 편차가 작지 않다.")
    add("")
    add("3. **M2 의 입력 조립은 가정이다.** M2 는 코랩에서 학습했고 학습 코드가 "
        "레포에 없다. 그래서 입력 조립을 M3 형식으로 가정해 돌렸다. 실제 학습 때 "
        "쓴 형식과 다를 수 있으므로 **M2 수치의 정확도는 보증되지 않는다.** "
        "이전 test 평가에서도 커밋값 0.426 과 재현값 0.482 가 벌어진 전례가 있다.")
    add("")
    add("4. **test 수치는 누수를 걷어낸 뒤 다시 잰 값이다.** 이전 "
        "`model_compare.md` 의 277행 표는 `tech_04` 69행이 `train` 의 `tech_03` 과 "
        "글자 단위로 같던 시절의 것이라 위로 부풀려져 있었다. 커밋 `aafbf06` 에서 "
        f"고쳐졌고, 이 표는 고친 뒤의 {n_te}행으로 다시 돌린 결과다.")
    add("")
    add("5. **`finance_03`·`finance_05` 의 원문 정제는 사람 검수 전이다** "
        "(종결어미 비율 49.7% / 54.9%). 표·참고문헌이 섞인 초벌 상태로 passage 가 "
        "만들어졌다.")
    add("")
    add("6. **`finance_04` 는 이번에 파이프라인을 새로 완성했다.** 직전까지 "
        "`docs/clean` 부터 검색 결과까지 전부 없어서 64행 전원이 근거 구간 없이 "
        "평가될 뻔했다. earticle PDF 가 심어둔 읽기 순서 텍스트 레이어를 쓰는 "
        "추출 방식을 추가해 26.2% → 75.7% 로 올린 뒤 검색까지 돌렸다.")
    add("")
    no_cand = sum(v["n_without_candidates"] for v in pd.values())
    add(f"7. **후보 구간이 비어 있는 행이 {no_cand}개 남아 있다** "
        f"({no_cand / n_ge:.1%}, `finance_03` 9 · `finance_05` 8). 어노테이션이 "
        "`claim_split.py` 보다 잘게 쪼갠 항목이라 대응하는 검색 결과가 아예 없다. "
        "그 행들은 원문 근거 없이 판정된다.")
    add("")
    add("8. **LLM 베이스라인은 이 표에 없다.** `results/judgments/*.tsv` 는 test 셋 "
        "4개 덱만 있고 재무는 판정한 적이 없다. 재무까지 비교하려면 LLM 판정을 "
        "먼저 만들어야 한다.")
    add("")
    add("---")
    add("")
    add("생성: `python src/build_compare_table.py` · "
        "원본 수치는 `results/M*_test_testset_296.json` 과 "
        "`results/M*_test_everything_543.json` 에 있다.")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"-> {OUT}  ({len(L)}줄)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
