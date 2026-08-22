#!/usr/bin/env python3
"""generalization(재무) 셋에서 M1 · M2 · M2' · M3 를 같은 자로 잰다.

    python src/eval_generalization.py

`eval_test_models.py` 의 짝이다. 그쪽은 test 셋 전용이라 여기서 재무를 맡는다.
추론 함수는 그쪽 것을 그대로 부른다 — 입력 조립을 다시 짜면 [SEP] 위치나
블록별 절단이 어긋나 성능이 낮게 나오고 원인을 나중에 못 찾는다.

출력은 `M3_test_everything.json` 과 같은 구조다 (runs[].generalization.*).
나중에 비교표에 그대로 합칠 수 있어야 하기 때문이다. 다만 **가중치가
모델당 하나뿐이라 runs 는 1개**다. 259행 결과는 시드 13/42/777 세 벌의
평균이었으므로 그 수치와 직접 비교하면 안 된다 — 아래 caveats 에 적어 둔다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_test_models as E  # noqa: E402

# dataset 표기(공백 있음) -> 결과 파일 표기(M3/M2' 형식). 259행 결과와 키를 맞춘다.
LABELS = ["근거 있음", "무근거", "모순", "Benign"]
OUT_LABELS = ["근거있음", "무근거", "모순", "benign"]
TO_OUT = dict(zip(LABELS, OUT_LABELS))

MODELS = [
    ("M1", None, None),          # frozen e5 + LinearSVC — 가중치가 레포에 있다
    ("M2", "M2", False),
    ("M2p", "M2p", False),
    ("M3", "M3", True),
]


def sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def block(gold: list[str], pred: list[str]) -> dict:
    """259행 결과와 같은 모양으로 만든다."""
    g = [TO_OUT[x] for x in gold]
    p = [TO_OUT[x] for x in pred]
    rep = classification_report(g, p, labels=OUT_LABELS, output_dict=True, zero_division=0)
    acc = rep["accuracy"]
    n = len(g)
    # 정확도의 정규근사 95% 구간. 259행 결과의 ci95 와 같은 의미로 쓴다.
    half = 1.96 * float(np.sqrt(acc * (1 - acc) / n)) if n else 0.0
    return {
        "macro_f1": f1_score(g, p, labels=OUT_LABELS, average="macro", zero_division=0),
        "accuracy": acc,
        "per_class_f1": {l: round(rep[l]["f1-score"], 4) for l in OUT_LABELS},
        "support": {l: int(rep[l]["support"]) for l in OUT_LABELS},
        "confusion": confusion_matrix(g, p, labels=OUT_LABELS).tolist(),
        "confusion_labels": OUT_LABELS,
        "n": n,
        "ci95": [max(0.0, acc - half), min(1.0, acc + half)],
    }


def per_deck(rows: list[dict], gold: list[str], pred: list[str]) -> dict:
    out: dict[str, dict] = {}
    for deck in sorted({r["deck_id"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["deck_id"] == deck]
        g, p = [gold[i] for i in idx], [pred[i] for i in idx]
        out[deck] = {
            "n": len(idx),
            "accuracy": sum(a == b for a, b in zip(g, p)) / len(idx),
            "macro_f1": f1_score([TO_OUT[x] for x in g], [TO_OUT[x] for x in p],
                                 labels=OUT_LABELS, average="macro", zero_division=0),
            "n_without_candidates": sum(1 for i in idx if not rows[i].get("candidates")),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--data", type=Path, default=Path("dataset/generalization.jsonl"))
    ap.add_argument("--suffix", default="_everything_543",
                    help="결과 파일명 꼬리. 259행 결과와 겹치지 않게 한다")
    args = ap.parse_args()

    raw = [json.loads(l) for l in args.data.open(encoding="utf-8") if l.strip()]
    rows = [r for r in raw if isinstance(r.get("label"), str) and r["label"] in LABELS]
    gold = [r["label"] for r in rows]
    print(f"{args.data}: {len(raw)}행 -> 라벨 유효 {len(rows)}행 "
          f"(라벨 미인식 {len(raw) - len(rows)}행 제외)")

    no_cand = sum(1 for r in rows if not r.get("candidates"))
    print(f"후보 구간이 없는 행 {no_cand}개 ({no_cand / len(rows):.1%})\n")

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    fingerprint = {
        n: {"sha12": sha12(args.root / "dataset" / f"{n}.jsonl"),
            "lines": sum(1 for _ in (args.root / "dataset" / f"{n}.jsonl").open(encoding="utf-8"))}
        for n in ("train", "test", "generalization")
    }

    caveats = [
        "가중치가 모델당 1개뿐이라 단일 seed 결과다. 259행 결과(M3/M2')는 "
        "seed 13/42/777 세 벌의 평균이므로 그 수치와 직접 비교하면 안 된다.",
        "평가 대상은 finance_01~05 전체다(543행 중 라벨 유효 538행). "
        "259행(sha12 4a3f556253bf)은 finance_01+02 만이었다.",
        f"후보 구간이 비어 있는 행이 {no_cand}개다. 어노테이션이 claim_split.py 보다 "
        "잘게 쪼갠 항목이라 대응하는 검색 결과가 없다. 그 행은 근거 없이 판정된다.",
        "finance_03·finance_05 의 docs/clean 은 사람 검수 전 초벌이다 "
        "(종결어미 비율 49.7% / 54.9%). finance_04 는 읽기레이어 추출로 75.7%.",
    ]

    summary = {}
    for model_id, mdir, use_ctx in MODELS:
        print(f"[{model_id}] 추론 중...")
        if model_id == "M1":
            pred = E.run_m1(args.root, rows)
            extra = {"backbone": "intfloat/multilingual-e5-large + LinearSVC",
                     "weights": "models/M1/classifier.joblib (레포에 있음)"}
            notes = list(caveats)
        else:
            w = args.root / "models" / mdir / "model.safetensors"
            if not w.exists():
                print(f"  건너뜀 — {w} 없음")
                continue
            pred = E.run_bert(args.root, args.root / "models" / mdir, rows, use_ctx)
            extra = {"backbone": "klue/roberta-base",
                     "weights": f"models/{mdir}/model.safetensors (.gitignore 대상)"}
            notes = list(caveats)
            if model_id == "M2":
                notes.insert(0, "M2 는 학습 코드가 레포에 없어 입력 조립을 M3 형식으로 "
                                "가정해 돌렸다. 실제 학습 때 쓴 형식과 다를 수 있으므로 "
                                "**이 수치의 정확도는 보증되지 않는다.** test 셋에서도 "
                                "커밋값 0.426 과 재현값 0.482 가 벌어진 전례가 있다.")

        g = block(gold, pred)
        doc = {
            "model_id": model_id,
            "evaluation_split": "generalization (finance_01~05)",
            "use_context": use_ctx,
            **extra,
            "single_seed": True,
            "seeds_note": "가중치 1벌만 확보되어 3-seed 평균·표준편차를 낼 수 없다",
            "data_fingerprint": fingerprint,
            "git_commit": commit,
            "runs": [{"seed": None, "generalization": g}],
            "generalization_macro_f1_mean": g["macro_f1"],
            "generalization_macro_f1_std": None,
            "per_deck": per_deck(rows, gold, pred),
            "caveats": notes,
        }
        out = args.root / "results" / f"{model_id}_test{args.suffix}.json"
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary[model_id] = g
        print(f"  macro-F1 {g['macro_f1']:.4f}  정확도 {g['accuracy']:.4f}  -> {out}")

    print("\n" + "=" * 62)
    print(f"{'모델':6}{'n':>6}{'macro-F1':>11}{'정확도':>10}   " + "  ".join(f"{l:>6}" for l in OUT_LABELS))
    for m, g in summary.items():
        cls = "  ".join(f"{g['per_class_f1'][l]:>6.3f}" for l in OUT_LABELS)
        print(f"{m:6}{g['n']:>6}{g['macro_f1']:>11.4f}{g['accuracy']:>10.4f}   {cls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
