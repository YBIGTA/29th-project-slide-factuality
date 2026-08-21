#!/usr/bin/env python3
"""test 셋에서 M1 · M2 · M2' · M3 · LLM 을 같은 자로 비교한다.

    python src/eval_test_models.py -o results/model_compare.md

M1 은 여기서 직접 돌린다 (frozen e5 + LinearSVC, 가중치가 레포에 있다).
M2 · M2' · M3 는 `models/**/model.safetensors` 가 .gitignore 에 걸려
레포에 없어서 다시 돌릴 수 없다. 각자 커밋해 둔 results/*.json 을 읽는다.

LLM 은 results/judgments/*.tsv — 원문 PDF 를 직접 읽고 붙인 판정이다.
claim 분할 단위가 dataset 과 달라(불릿을 쪼갠다) 원래 claim 으로 되묶는다.
한 claim 안에 여러 주장이 있으면 **가장 나쁜 것**을 그 claim 의 라벨로 삼는다
— 세 줄 중 한 줄이 무근거면 그 상자를 그대로 믿을 수 없기 때문이다.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_deck_claims import parts as split_parts

LABELS = ["근거 있음", "무근거", "모순", "Benign"]
SEVERITY = {"모순": 3, "무근거": 2, "근거 있음": 1, "Benign": 0}   # 되묶을 때 큰 쪽이 이긴다
NAN = float("nan")


def norm(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def read_tsv(p: Path) -> list[list[str]]:
    return [l.rstrip("\n").split("\t") for l in p.open(encoding="utf-8") if l.strip()]


# ── LLM 판정을 dataset 의 claim 단위로 되묶는다 ────────────────────────
def llm_by_claim(root: Path, judgments: Path) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for tsv in sorted(judgments.glob("*.tsv")):
        if tsv.name.endswith(".extra.tsv"):
            continue
        doc_id = tsv.stem
        deck = sorted((root / "claims").glob(doc_id + "__*.jsonl"))[0]
        verdict = {r[0]: r[1] for r in read_tsv(tsv)}

        per: dict[int, int] = {}
        for row in read_jsonl(deck):
            key = "Claim (PPT)" if "Claim (PPT)" in row else "claim_text"
            slide = int(row.get("Slide #") or row.get("slide"))
            labels = []
            for _ in split_parts(str(row[key])):
                per[slide] = per.get(slide, 0) + 1
                labels.append(verdict.get("s%02d_c%02d" % (slide, per[slide])))
            labels = [l for l in labels if l]
            if labels:
                out[(doc_id, norm(row[key]))] = max(labels, key=lambda l: SEVERITY[l])
    return out


# ── M1 을 실제로 돌린다 ──────────────────────────────────────────────
def run_m1(root: Path, rows: list[dict]) -> list[str]:
    import joblib
    from sentence_transformers import SentenceTransformer
    from train_m1 import build_features, candidate_texts, encode_lookup

    # joblib 안은 분류기 자체가 아니라 라벨 순서까지 같이 담은 dict 다
    bundle = joblib.load(root / "models" / "M1" / "classifier.joblib")
    clf, labels = bundle["classifier"], bundle["labels"]
    enc = SentenceTransformer(bundle["encoder"])
    dim = int(bundle["embedding_dimension"])

    claims = [str(r.get("claim_text", "")).strip() for r in rows]
    passages = [t for r in rows for t in candidate_texts(r)]
    x = build_features(rows, encode_lookup(enc, claims, "query: ", 32),
                       encode_lookup(enc, passages, "passage: ", 32), dim)
    return [labels[i] for i in clf.predict(x)]


def score(gold: list[str], pred: list[str]) -> dict:
    rep = classification_report(gold, pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "n": len(gold),
        "macro_f1": f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0),
        "accuracy": rep["accuracy"],
        "per_class": {l: rep[l]["f1-score"] for l in LABELS},
        "support": {l: int(rep[l]["support"]) for l in LABELS},
        "confusion": confusion_matrix(gold, pred, labels=LABELS).tolist(),
    }


def from_json(path: Path) -> dict | None:
    """팀원이 커밋한 결과 json 에서 필요한 값만 꺼낸다. 형식이 두 가지다."""
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    if "classification_report" in d:                       # M1 · M2 형식
        rep = d["classification_report"]
        return {"macro_f1": d["macro_f1"], "accuracy": d["accuracy"],
                "per_class": {l: rep[l]["f1-score"] for l in LABELS if l in rep},
                "seeds": 1}
    if "runs" in d:                                        # M2' · M3 형식 (시드 3개)
        runs = [r["test"] for r in d["runs"]]
        # M2'·M3 는 키를 공백 없이 · Benign 은 소문자로 쓴다
        keys = {"근거 있음": "근거있음", "무근거": "무근거", "모순": "모순", "Benign": "benign"}
        return {"macro_f1": float(np.mean([r["macro_f1"] for r in runs])),
                "std": float(np.std([r["macro_f1"] for r in runs])),
                "accuracy": float(np.mean([r["accuracy"] for r in runs])),
                "per_class": {l: float(np.mean([r["per_class_f1"].get(k, 0.0) for r in runs]))
                              for l, k in keys.items()},
                "seeds": len(runs)}
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--judgments", type=Path, default=Path("results/judgments"))
    ap.add_argument("-o", "--out", type=Path, default=Path("results/model_compare.md"))
    ap.add_argument("--skip-m1", action="store_true")
    args = ap.parse_args()

    rows = [r for r in read_jsonl(args.root / "dataset" / "test.jsonl") if r.get("label") in SEVERITY]
    gold = [r["label"] for r in rows]
    print("test %d행 (라벨이 4분류인 것만; 원본 290행)" % len(rows))
    with_cand = sum(1 for r in rows if r.get("candidates"))
    print("  후보 구간이 붙어 있는 행: %d (%.0f%%)" % (with_cand, 100 * with_cand / len(rows)))

    results: dict[str, dict] = {}

    # 학습셋과 claim 본문이 그대로 겹치는 행 — 모델이 외운 것이라 성능이 아니다
    train_claims = {norm(r["claim_text"])
                    for r in read_jsonl(args.root / "dataset" / "train.jsonl")}
    leaked = [norm(r["claim_text"]) in train_claims for r in rows]
    by_deck: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_deck.setdefault(str(r["doc_id"]), []).append(i)
    print("  train 과 claim 이 그대로 겹치는 행: %d" % sum(leaked))
    for d, idx in sorted(by_deck.items()):
        n = sum(leaked[i] for i in idx)
        if n:
            print("    %s  %d/%d" % (d, n, len(idx)))

    # LLM — 판정한 덱만 센다. 대응 안 되는 행을 최빈값으로 메우면
    # 실제로 판정하지 않은 것을 맞힌 셈이 되어 수치가 뒤틀린다.
    table = llm_by_claim(args.root, args.judgments)
    llm = [table.get((str(r["doc_id"]), norm(r["claim_text"]))) for r in rows]
    for d, idx in sorted(by_deck.items()):
        got = sum(llm[i] is not None for i in idx)
        print("  LLM 대응 %s: %d/%d" % (d, got, len(idx)))

    # M1 — 여기서 직접 돌린다
    m1 = None
    if not args.skip_m1:
        m1 = run_m1(args.root, rows)
        results["M1 (frozen e5 + LinearSVC)"] = score(gold, m1)

    for name, f in [("M2 (KLUE-RoBERTa, 맥락 없음)", "M2_test.json"),
                    ("M2' (동일 백본 대조군)", "M2p_test.json"),
                    ("M3 (KLUE-RoBERTa, 맥락 포함)", "M3_test.json")]:
        got = from_json(args.root / "results" / f)
        if got:
            results[name] = got

    # ── 표 만들기 ────────────────────────────────────────────────
    def table(rows_out, title, note):
        out = ["## " + title, "", note, "",
               "| 모델 | n | macro-F1 | 정확도 | 근거 있음 | 무근거 | 모순 | Benign |",
               "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        best = max((r["macro_f1"] for _, r in rows_out), default=0)
        for name, r in sorted(rows_out, key=lambda kv: -kv[1]["macro_f1"]):
            f1 = ("**%.3f**" if r["macro_f1"] == best else "%.3f") % r["macro_f1"]
            if r.get("seeds", 1) > 1:
                f1 += " ±%.3f" % r["std"]
            pc = r["per_class"]
            out.append("| %s | %s | %s | %.3f | " % (name, r.get("n", "—"), f1, r["accuracy"]) +
                       " | ".join("%.3f" % pc.get(l, NAN) for l in LABELS) + " |")
        return out + [""]

    lines = ["# test 셋 모델 비교", "",
             "덱 4개 (`arts_03` · `bio_04` · `socio_04` · `tech_04`) · claim %d개" % len(rows),
             "라벨이 4분류가 아닌 13행(빈칸 12 + 근거문이 라벨칸에 들어간 1행)은 뺐다.", ""]

    lines += table(list(results.items()), "전체 277행",
                   "팀원들이 각자 낸 수치와 같은 조건이다. **아래 누수 항목을 먼저 읽을 것.**")

    # 누수
    lines += ["## 이 표를 그대로 믿으면 안 되는 이유", "",
              "`dataset/test.jsonl` 의 `tech_04` 69행은 **claim 본문이 `train.jsonl` 의 "
              "`tech_03` 69행과 글자 단위로 같다.** 내용도 tech_03 원문(STRM 보안 아키텍처)이고, "
              "레포의 실제 `tech_04`(연소 불안정)와는 다른 논문이다. doc_id 가 잘못 붙었다.", "",
              "모델은 이 69행을 학습 때 이미 봤다. M1 의 덱별 정확도가 그것을 보여준다.", "",
              "| 덱 | M1 정확도 | |", "| --- | ---: | --- |"]
    per_deck = json.loads((args.root / "results" / "M1_test.json").read_text(encoding="utf-8")).get("per_deck", {})
    for d, v in sorted(per_deck.items()):
        mark = " **← 학습에서 본 것**" if d == "tech_04" else ""
        lines.append("| %s | %.3f |%s |" % (d, v["accuracy"], mark))
    lines += ["", "test 277행 중 **69행(25%)** 이 여기 해당한다. "
              "`arts_03` 은 또 다른 문제로, 라벨이 붙은 덱과 지금 레포에 올라온 "
              "`arts_03__gamma.pptx` 가 서로 다른 판본이다 (claim 일치율 41%).", ""]

    # 누수 제거 + LLM 대응되는 행만
    keep = [i for i in range(len(rows)) if not leaked[i] and llm[i] is not None]
    if keep and m1 is not None:
        g = [gold[i] for i in keep]
        sub = [("M1 (frozen e5 + LinearSVC)", score(g, [m1[i] for i in keep])),
               ("LLM (원문 직접 판정)", score(g, [llm[i] for i in keep]))]
        decks = sorted({str(rows[i]["doc_id"]) for i in keep})
        lines += table(sub, "누수 제거 · 정면 비교",
                       "학습에서 본 행을 빼고, LLM 이 실제로 판정한 행만 남긴 %d행 (%s). "
                       "M2·M2'·M3 는 행 단위 예측이 없어 이 표에 못 넣는다."
                       % (len(keep), " · ".join(decks)))

    sup = score(gold, gold)["support"]
    lines += ["정답 분포(전체 277행): " + " · ".join("%s %d" % (l, sup[l]) for l in LABELS), ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(chr(10).join(lines) + chr(10), encoding="utf-8")
    args.out.with_suffix(".json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(chr(10) + chr(10).join(lines))
    print(chr(10) + "저장: %s" % args.out)


if __name__ == "__main__":
    main()
