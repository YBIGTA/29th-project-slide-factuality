#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/check_token_budget.py — 512가 실제로 걸리는지 데이터로 확인한다.

절단 설정을 정하기 전에 이걸 먼저 돌려라. "후보가 길어서 claim이 밀린다"는
추측이고, 이 스크립트가 내놓는 건 숫자다. 절단이 필요 없는데 64토큰으로
자르고 있으면 그냥 정보를 버리는 것이다.

python src/check_token_budget.py --data dataset/train.jsonl
"""
import argparse
import json
from collections import Counter

import numpy as np
from transformers import AutoTokenizer


def pct(a, ps=(50, 90, 95, 99, 100)):
    return {f"p{p}": int(np.percentile(a, p)) for p in ps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/train.jsonl")
    ap.add_argument("--backbone", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    ap.add_argument("--n_cand", type=int, default=5)
    ap.add_argument("--ctx_bullets", type=int, default=6)
    ap.add_argument("--max_len", type=int, default=512)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.backbone)
    n = lambda s: len(tok(str(s or ""), add_special_tokens=False)["input_ids"])

    recs = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    claim, ctx, cand_each, total_m3, total_m2 = [], [], [], [], []

    for r in recs:
        c = n(r["claim_text"])
        parts = [r.get("slide_title") or ""] + [str(b) for b in (r.get("slide_context") or [])][: a.ctx_bullets]
        x = n(" / ".join(p for p in parts if p))
        cs = [n(k.get("text", "")) for k in (r.get("candidates") or [])[: a.n_cand]]
        claim.append(c); ctx.append(x); cand_each += cs
        total_m3.append(c + x + sum(cs) + 3 + len(cs))
        total_m2.append(c + sum(cs) + 2 + len(cs))

    print(f"{a.data} — {len(recs)} claims, 토크나이저 model_max_length={tok.model_max_length}\n")
    for name, arr in [("claim", claim), ("슬라이드 맥락", ctx),
                      ("후보 1개", cand_each), ("M3 전체 (절단 전)", total_m3),
                      ("M2 전체 (절단 전)", total_m2)]:
        print(f"{name:22s} {pct(arr)}  평균 {np.mean(arr):.0f}")

    over3 = np.mean([t > a.max_len for t in total_m3])
    over2 = np.mean([t > a.max_len for t in total_m2])
    print(f"\n{a.max_len} 초과 비율 — M3 {over3:.1%} / M2 {over2:.1%}")

    if over3 == 0:
        print("→ 절단 불필요. cand_max 를 크게 잡아 정보를 다 넣어라.")
    else:
        # 후보에 남겨줄 수 있는 예산에서 역산
        c95, x95 = np.percentile(claim, 95), np.percentile(ctx, 95)
        for k in (5, 4, 3):
            budget = (a.max_len - 8 - c95 - x95) / k
            print(f"→ n_cand={k} 이면 cand_max ≈ {int(budget)} "
                  f"(claim p95={int(c95)}, 맥락 p95={int(x95)} 확보 기준)")

    # M2/M3 공정성 확인: M2가 절단을 덜 당하면 비교가 M3에 불리해진다
    if over3 > over2 + 0.05:
        print("\n⚠ M3가 M2보다 절단을 훨씬 많이 당한다. 맥락을 넣은 효과와 "
              "'후보가 잘려서 손해본 효과'가 섞인다. 두 모델의 cand_max 를 같게 "
              "고정하고(둘 다 절단되게) 비교할 것.")

    print("\n라벨 분포:", Counter(r["label"] for r in recs))


if __name__ == "__main__":
    main()
