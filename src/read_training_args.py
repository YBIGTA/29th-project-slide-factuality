#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/read_training_args.py — HF Trainer 가 저장한 training_args.bin 을 사람이 읽게 출력한다.

python src/read_training_args.py models/M2/training_args.bin
python src/read_training_args.py models/M2/training_args.bin models/M3/training_args.bin  # 두 개면 diff
"""
import sys
from pathlib import Path

import torch

# 판정 모델 결과를 좌우하는 것들만 추려서 먼저 보여준다
KEY = ["seed", "learning_rate", "num_train_epochs", "per_device_train_batch_size",
       "gradient_accumulation_steps", "per_device_eval_batch_size", "weight_decay",
       "warmup_ratio", "warmup_steps", "max_grad_norm", "label_smoothing_factor",
       "lr_scheduler_type", "optim", "fp16", "bf16", "eval_strategy",
       "evaluation_strategy", "save_strategy", "load_best_model_at_end",
       "metric_for_best_model", "greater_is_better", "num_train_epochs"]


def load(p):
    o = torch.load(p, map_location="cpu", weights_only=False)
    return o.to_dict() if hasattr(o, "to_dict") else dict(vars(o))


def fmt(v):
    return str(getattr(v, "value", v))


def main():
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        sys.exit(1)

    dicts = []
    for p in paths:
        if not Path(p).exists():
            print(f"없음: {p}")
            sys.exit(1)
        dicts.append(load(p))

    if len(dicts) == 1:
        d = dicts[0]
        print(f"\n{paths[0]}\n" + "=" * 70)
        seen = []
        for k in KEY:
            if k in d and k not in seen:
                seen.append(k)
                print(f"  {k:<34} {fmt(d[k])}")
        eff = (d.get("per_device_train_batch_size", 1)
               * d.get("gradient_accumulation_steps", 1))
        print(f"  {'(유효 배치)':<34} {eff}")
        print("\n  --- 그 외 기본값과 다른 항목 ---")
        for k in sorted(d):
            if k not in seen and d[k] not in (None, False, 0, 0.0, "", [], {}, "no"):
                print(f"  {k:<34} {fmt(d[k])}")
        print("\n이 값들을 M3 에 그대로 물려주려면:")
        print(f"  python src/train_judge_m3.py --model_id M3 --use_context \\")
        print(f"         --backbone klue/roberta-base --match_args {paths[0]}")
        return

    # 두 개 이상이면 차이만
    a, b = dicts[0], dicts[1]
    print(f"\nA = {paths[0]}\nB = {paths[1]}\n" + "=" * 70)
    diff = [k for k in sorted(set(a) | set(b))
            if a.get(k) != b.get(k) and k not in ("output_dir", "logging_dir", "run_name",
                                                  "hub_model_id", "_n_gpu")]
    if not diff:
        print("  차이 없음 ✅")
        return
    print(f"  {'항목':<34} {'A':<22} B")
    for k in diff:
        print(f"  {k:<34} {fmt(a.get(k)):<22} {fmt(b.get(k))}")
    print(f"\n  → {len(diff)}개 항목이 다르다. ablation 짝이라면 이 목록이 비어 있어야 한다.")


if __name__ == "__main__":
    main()
