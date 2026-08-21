#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/check_ablation.py — ablation 짝이 정말 "플래그 하나만" 다른지 기계로 검증한다.

"같은 조건으로 돌렸습니다"는 기억이고, 이건 증거다. 최종 발표 전에 돌려서
통과 로그를 슬라이드 부록에 넣으면 "변수 뭘로 통제했냐" 질문이 끝난다.

python src/check_ablation.py --base results/M2p_test.json --variant results/M3_test.json
"""
import argparse
import json
import sys
from pathlib import Path

# 이 키들은 달라도 정상이다 (실험 조건이 아니라 이름·경로·의도된 조작)
ALLOWED = {"model_id", "model_dir", "out_dir", "use_context"}

# 이 키들은 달라지면 ablation 자체가 무효
CRITICAL = {"backbone", "arch", "seeds", "max_len", "claim_max", "n_cand", "cand_max",
            "lr", "head_lr", "epochs", "patience", "batch_size", "grad_accum",
            "weight_decay", "warmup_ratio", "dropout", "label_smoothing", "cw_power",
            "span_head", "span_loss_w", "n_dev_docs", "data_dir"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="맥락 없는 쪽 (M2')")
    ap.add_argument("--variant", required=True, help="맥락 있는 쪽 (M3)")
    a = ap.parse_args()

    A = json.loads(Path(a.base).read_text(encoding="utf-8"))
    B = json.loads(Path(a.variant).read_text(encoding="utf-8"))
    ca, cb = A["config"], B["config"]
    fail, warn = [], []

    # 1) use_context 가 실제로 다른가
    if not (ca.get("use_context") is False and cb.get("use_context") is True):
        fail.append(f"use_context 가 (False, True) 가 아니다: "
                    f"{ca.get('use_context')} / {cb.get('use_context')}")

    # 2) 그 외에 다른 게 있는가
    diff = sorted(k for k in set(ca) | set(cb)
                  if ca.get(k) != cb.get(k) and k not in ALLOWED)
    for k in diff:
        msg = f"{k}: {ca.get(k)!r}  vs  {cb.get(k)!r}"
        (fail if k in CRITICAL else warn).append(msg)

    # 2b) 실제로 Trainer 에 들어간 하이퍼파라미터가 같은가
    #     (--match_args 를 한쪽만 썼으면 config 는 비슷해도 이 값이 갈린다)
    ha, hb = A.get("hyperparameters") or {}, B.get("hyperparameters") or {}
    if ha or hb:
        for k in sorted(set(ha) | set(hb)):
            if ha.get(k) != hb.get(k):
                fail.append(f"hyperparameters.{k}: {ha.get(k)!r}  vs  {hb.get(k)!r}")
    else:
        warn.append("hyperparameters 없음 — Trainer 설정 동일성 증명 불가")

    # 3) 데이터 스냅샷이 같은가 — 가장 놓치기 쉬운 항목
    fa, fb = A.get("data_fingerprint"), B.get("data_fingerprint")
    if not fa or not fb:
        warn.append("data_fingerprint 없음 (구버전 스크립트 결과) — 데이터 동일성 증명 불가")
    elif fa != fb:
        for name in sorted(set(fa) | set(fb)):
            if fa.get(name) != fb.get(name):
                fail.append(f"데이터 불일치 {name}: {fa.get(name)}  vs  {fb.get(name)}")

    # 4) 시드별 dev 분할이 같은가 (같은 시드면 같은 덱이 dev로 빠져야 정상)
    ra = {r["seed"]: r.get("dev_docs") for r in A.get("runs", [])}
    rb = {r["seed"]: r.get("dev_docs") for r in B.get("runs", [])}
    if set(ra) != set(rb):
        fail.append(f"시드 집합 불일치: {sorted(ra)} vs {sorted(rb)}")
    for s in sorted(set(ra) & set(rb)):
        if ra[s] != rb[s]:
            fail.append(f"seed {s} 의 dev 덱이 다르다: {ra[s]} vs {rb[s]}")

    # 5) 절단 공정성 — 맥락이 들어간 쪽만 후보가 더 잘리면 교란요인
    if ca.get("cand_max") == cb.get("cand_max") and ca.get("n_cand") == cb.get("n_cand"):
        pass
    else:
        fail.append("후보 예산(n_cand/cand_max)이 다르다 — 맥락 효과와 절단 손실이 섞인다")

    # ---- 보고 ----
    print(f"base    : {a.base}   ({A.get('model_id')}, context={ca.get('use_context')})")
    print(f"variant : {a.variant}   ({B.get('model_id')}, context={cb.get('use_context')})")
    print(f"git     : {A.get('git_commit')} / {B.get('git_commit')}\n")

    for m in warn:
        print(f"  ⚠  {m}")
    for m in fail:
        print(f"  ✗  {m}")

    if fail:
        print(f"\n❌ ablation 무효 — 위 {len(fail)}건을 맞추고 재실행할 것.")
        print("   두 조건은 같은 사람이 같은 스크립트로 연달아 돌려야 한다.")
        sys.exit(1)

    # 통과했으면 효과 크기를 표준편차와 함께 보고
    ma, sa = A.get("test_macro_f1_mean"), A.get("test_macro_f1_std")
    mb, sb = B.get("test_macro_f1_mean"), B.get("test_macro_f1_std")
    print("✅ use_context 외에 다른 조건 차이 없음. ablation 유효.\n")
    if None not in (ma, mb):
        d = mb - ma
        pooled = ((sa or 0) ** 2 + (sb or 0) ** 2) ** 0.5
        print(f"   맥락 없음 (M2')  macro-F1  {ma:.4f} ± {sa:.4f}")
        print(f"   맥락 포함 (M3)   macro-F1  {mb:.4f} ± {sb:.4f}")
        print(f"   차이             {d:+.4f}   (시드 분산 합성 {pooled:.4f})")
        if abs(d) <= pooled:
            print("\n   ⚠ 차이가 시드 분산 안에 들어온다. '맥락이 효과 있다'고 주장하지 말고")
            print("     '측정된 차이가 시드 분산을 넘지 않았다'로 보고하는 것이 방어 가능하다.")
        else:
            print("\n   → 차이가 시드 분산을 넘는다. 맥락 효과 주장 가능.")

        # 시드별 짝 비교 — 평균만 보면 정보를 버린다
        pa = {r["seed"]: r["test"]["macro_f1"] for r in A.get("runs", []) if "test" in r}
        pb = {r["seed"]: r["test"]["macro_f1"] for r in B.get("runs", []) if "test" in r}
        common = sorted(set(pa) & set(pb))
        if common:
            print("\n   시드별 짝 비교:")
            for s in common:
                print(f"     seed {s:<5} {pa[s]:.4f} → {pb[s]:.4f}  ({pb[s]-pa[s]:+.4f})")
            wins = sum(pb[s] > pa[s] for s in common)
            print(f"     M3 우세 {wins}/{len(common)} 시드")


if __name__ == "__main__":
    main()
