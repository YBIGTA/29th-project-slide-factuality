#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/train_judge_m3.py — M3 (맥락 포함) / M2' (맥락 없음) 학습 · 저장 · 리포트

시나님 M2와 같은 폴더 구성으로 저장한다 (HF Trainer 표준 출력):
    models/M3/
      ├─ M3_README.md          ← 하이퍼파라미터·데이터 지문·결과 전부 기록 (Q&A 대비)
      ├─ config.json
      ├─ tokenizer.json
      ├─ tokenizer_config.json
      ├─ special_tokens_map.json
      ├─ training_args.bin
      └─ model.safetensors     ← 용량 커서 git 에는 안 올라감 (.gitignore)
    results/M3_test.json

사용법
------
# 1) 토큰 예산 먼저 확인
python src/check_token_budget.py --data dataset/train.jsonl --backbone klue/roberta-base

# 2) M3 (맥락 포함) — 시나님 M2 하이퍼파라미터를 그대로 채택
python src/train_judge_m3.py --model_id M3  --use_context \
       --backbone klue/roberta-base --match_args models/M2/training_args.bin

# 3) M2' (맥락 없음) — 위와 완전히 동일, --use_context 만 뺀다
python src/train_judge_m3.py --model_id M2p \
       --backbone klue/roberta-base --match_args models/M2/training_args.bin

# 4) 두 결과가 정말 플래그 하나만 다른지 기계 검증
python src/check_ablation.py --base results/M2p_test.json --variant results/M3_test.json
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import random
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)

LABELS = ["근거있음", "무근거", "모순", "benign"]
L2I = {l: i for i, l in enumerate(LABELS)}
_ALIAS = {"근거있음": "근거있음", "근거 있음": "근거있음", "근거있음 ": "근거있음",
          "supported": "근거있음", "근거 존재": "근거있음",
          "무근거": "무근거", "unsupported": "무근거", "근거없음": "무근거", "근거 없음": "무근거",
          "모순": "모순", "contradiction": "모순", "contradicted": "모순", "충돌": "모순",
          "benign": "benign", "benign_true": "benign", "상식참": "benign"}


def norm_label(s):
    """알 수 없는 라벨은 None 을 돌려준다 (예외를 던지지 않는다).
    763건 중 일부가 비어 있다는 보고가 있어, 한 행 때문에 학습이 죽지 않게 한다."""
    s = str(s or "").strip()
    return _ALIAS.get(s) or _ALIAS.get(s.lower())


# ----------------------------------------------------------------------------
# 필드명 정규화 — 레포에 계획서 스키마와 은채님 스프레드시트 형식이 섞여 있다
# ----------------------------------------------------------------------------
FIELD = {
    "claim_text":    ["claim_text", "Claim (PPT)", "claim", "Claim", "주장 텍스트", "text"],
    "label":         ["label", "Label", "라벨", "gold", "판정"],
    "doc_id":        ["doc_id", "docid", "document_id"],
    "deck_id":       ["deck_id", "deckid"],
    "claim_id":      ["claim_id", "claimid", "id"],
    "slide_title":   ["slide_title", "Slide_Title", "Slide Title", "슬라이드 제목", "슬라이드명"],
    "slide_context": ["slide_context", "Context (PPT)", "context", "Context", "맥락"],
    "candidates":    ["candidates", "cands", "top5", "retrieved", "candidate_spans"],
    "evidence_span": ["evidence_span", "evidence_span_id", "gold_span", "근거 구간"],
}
CAND_TEXT = ["text", "span_text", "passage", "sent", "구간", "content"]
CAND_ID = ["span_id", "id", "sid", "sent_id"]


def get(rec, canon, default=None):
    for k in FIELD[canon]:
        if k in rec and rec[k] not in (None, ""):
            return rec[k]
    return default


def normalize(recs, verbose=True):
    """레코드를 내부 표준형으로 바꾸고, 못 쓰는 행은 세어서 버린다."""
    out, drop_label, drop_claim, no_cand = [], 0, 0, 0
    for r in recs:
        claim = get(r, "claim_text")
        lab = norm_label(get(r, "label"))
        if not claim:
            drop_claim += 1
            continue
        if lab is None:
            drop_label += 1
            continue

        ctx = get(r, "slide_context", [])
        if isinstance(ctx, str):
            ctx = [ctx]

        cands = []
        for c in (get(r, "candidates", []) or []):
            if isinstance(c, str):
                cands.append({"text": c, "span_id": None})
            else:
                cands.append({
                    "text": next((c[k] for k in CAND_TEXT if c.get(k)), ""),
                    "span_id": next((c[k] for k in CAND_ID if c.get(k) is not None), None)})
        if not cands:
            no_cand += 1

        doc = get(r, "doc_id") or (str(get(r, "deck_id", "")).split("__")[0] or None)
        out.append({"claim_text": str(claim), "label": lab,
                    "slide_title": str(get(r, "slide_title", "") or ""),
                    "slide_context": [str(x) for x in ctx],
                    "candidates": cands, "evidence_span": get(r, "evidence_span"),
                    "doc_id": doc or "UNKNOWN_DOC",
                    "claim_id": get(r, "claim_id", f"row{len(out)}")})

    if verbose:
        print(f"  정규화: {len(recs)} → {len(out)} 행 사용 "
              f"(claim 없음 {drop_claim} / 라벨 미인식·공백 {drop_label} 버림)")
        if no_cand:
            print(f"  ⚠ 후보 구간이 없는 행 {no_cand}개 — 그 행은 원문 근거 없이 학습된다")
        if any(r["doc_id"] == "UNKNOWN_DOC" for r in out):
            print("  ⚠ doc_id 를 못 찾은 행이 있다 — 문서 단위 dev 분할이 부정확해진다")
    return out


# ============================================================================
# 1. 입력 조립 — ★ M3 의 본체 ★
# ============================================================================
# [CLS] claim [SEP] 슬라이드 제목 + 같은 슬라이드 다른 불릿 [SEP] 후보1 [SEP] ... 후보5 [SEP]
#         ↑검증 대상     ↑ --use_context 일 때만 (M3 와 M2' 의 유일한 차이)      ↑검색기 top-5
#
# 블록마다 따로 자른다. HF 기본 truncation("longest_first")에 맡기면 어느 블록이
# 희생되는지 통제할 수 없고, claim 이 밀려 나가면 M3 가 M2' 보다 낮게 나온다.
#
# klue/roberta-base 는 절대 위치 임베딩이라 상한을 넘기면 크래시한다.
# 실사용 상한이 512 가 아니라 510~511 이므로 --max_len 510 을 기본값으로 둔다.

def enc(tok, text, cap):
    if not text:
        return []
    return tok(str(text), add_special_tokens=False)["input_ids"][:cap]


def build_one(rec, tok, a):
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    ids = [cls_id] + enc(tok, rec["claim_text"], a.claim_max) + [sep_id]

    if a.use_context:
        parts = [rec["slide_title"]] + rec["slide_context"][: a.ctx_bullets]
        ids += enc(tok, " / ".join(p for p in parts if p), a.ctx_max) + [sep_id]

    for c in rec["candidates"][: a.n_cand]:
        cids = enc(tok, c["text"], a.cand_max)
        if cids:
            ids += cids + [sep_id]

    ids = ids[: a.max_len]
    return {"input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": L2I[rec["label"]]}


class JudgeDS(Dataset):
    def __init__(self, recs, tok, a):
        self.rows = [build_one(r, tok, a) for r in recs]
        self.docs = [r["doc_id"] for r in recs]
        self.lens = [len(x["input_ids"]) for x in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class PadCollator:
    """★ 배치 안에서 가장 긴 것에 맞춰 패딩한다.
    이게 없으면 길이가 다른 행을 텐서로 못 만들어
    `ValueError: expected sequence of length ...` 로 죽는다.
    (Trainer 기본 collator 는 transformers 버전에 따라 패딩을 안 한다.)"""

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        L = max(len(f["input_ids"]) for f in feats)
        return {
            "input_ids": torch.tensor(
                [f["input_ids"] + [self.pad_id] * (L - len(f["input_ids"])) for f in feats],
                dtype=torch.long),
            "attention_mask": torch.tensor(
                [f["attention_mask"] + [0] * (L - len(f["attention_mask"])) for f in feats],
                dtype=torch.long),
            "labels": torch.tensor([f["labels"] for f in feats], dtype=torch.long),
        }


# ============================================================================
# 2. 클래스 가중치 Trainer
# ============================================================================
# 근거있음이 60~70% 인 데이터에서 가중치를 안 주면 macro-F1 이 크게 떨어진다.
# 이 설정이 M3 와 M2' 에서 반드시 같아야 한다 (다르면 맥락 효과와 섞인다).

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kw):
        super().__init__(*args, **kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        out = model(**inputs)
        w = self.class_weights.to(out.logits.device) if self.class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=w, label_smoothing=self.args.label_smoothing_factor)(
            out.logits.view(-1, len(LABELS)), labels.view(-1))
        inputs["labels"] = labels
        return (loss, out) if return_outputs else loss


def class_weights(recs, power):
    cnt = Counter(L2I[r["label"]] for r in recs)
    tot = sum(cnt.values())
    w = [(tot / max(cnt.get(i, 0), 1)) ** power for i in range(len(LABELS))]
    m = float(np.mean(w))
    return torch.tensor([x / m for x in w], dtype=torch.float)


# ============================================================================
# 3. 지표
# ============================================================================
def metrics_from(y_true, y_pred):
    f1s, sup = [], []
    for c in range(len(LABELS)):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
        sup.append(tp + fn)
    cm = [[0] * len(LABELS) for _ in LABELS]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return {"macro_f1": float(np.mean(f1s)),
            "accuracy": float(np.mean([t == p for t, p in zip(y_true, y_pred)])),
            "per_class_f1": dict(zip(LABELS, [round(x, 4) for x in f1s])),
            "support": dict(zip(LABELS, sup)),
            "confusion": cm, "confusion_labels": LABELS, "n": len(y_true)}


def compute_metrics(ev):
    logits = ev.predictions[0] if isinstance(ev.predictions, tuple) else ev.predictions
    return {"macro_f1": metrics_from(list(ev.label_ids), list(np.argmax(logits, -1)))["macro_f1"]}


def doc_bootstrap_ci(y_true, y_pred, docs, n_boot=1000, seed=0):
    """문서 단위 부트스트랩. 같은 문서의 claim 은 독립이 아니라서 claim 단위로
    리샘플링하면 신뢰구간이 실제보다 좁게 나온다."""
    rng = np.random.default_rng(seed)
    by = {}
    for i, d in enumerate(docs):
        by.setdefault(d, []).append(i)
    keys = list(by)
    vals = []
    for _ in range(n_boot):
        idx = [i for k in rng.choice(len(keys), len(keys), replace=True) for i in by[keys[k]]]
        vals.append(metrics_from([y_true[i] for i in idx], [y_pred[i] for i in idx])["macro_f1"])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


# ============================================================================
# 4. 데이터
# ============================================================================
def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def fingerprint(data_dir):
    """M3 와 M2' 가 같은 데이터 스냅샷으로 돌았는지 증명하는 유일한 수단.
    dataset/ 은 8/19~8/21 계속 바뀌었으므로 사람 기억으로는 못 잡는다."""
    out = {}
    for n in ("train.jsonl", "test.jsonl", "generalization.jsonl"):
        p = Path(data_dir) / n
        if p.exists():
            b = p.read_bytes()
            out[n] = {"sha12": hashlib.sha256(b).hexdigest()[:12], "lines": b.count(b"\n")}
    return out


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def split_dev_by_doc(recs, n_dev, seed):
    """★ 문서 단위로만 나눈다. claim 단위 랜덤 분할은 즉시 데이터 누수다
    (같은 문서의 claim 들이 근거 구간을 공유하므로)."""
    docs = sorted({r["doc_id"] for r in recs})
    rng = random.Random(seed)
    rng.shuffle(docs)
    dev = set(docs[:n_dev])
    return ([r for r in recs if r["doc_id"] not in dev],
            [r for r in recs if r["doc_id"] in dev], sorted(dev))


# ============================================================================
# 5. transformers 버전 호환 (Colab 4.x / 로컬 5.x 둘 다 돌게)
# ============================================================================
def make_targs(**kw):
    sig = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in sig and "eval_strategy" in kw:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    return TrainingArguments(**{k: v for k, v in kw.items() if k in sig})


def make_trainer(cls, **kw):
    sig = inspect.signature(Trainer.__init__).parameters
    tok = kw.pop("tokenizer", None)
    if tok is not None:
        kw["processing_class" if "processing_class" in sig else "tokenizer"] = tok
    return cls(**kw)


def read_match_args(path):
    """시나님 training_args.bin 을 읽어 하이퍼파라미터를 그대로 채택한다.
    그러면 M2' 가 시나님 M2 와 거의 같은 조건이 되고, M3 는 M2' 와 정확히 같아진다.
    = 시나님 작업을 버리지 않으면서 ablation 을 성립시키는 방법."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    d = obj.to_dict() if hasattr(obj, "to_dict") else dict(vars(obj))
    keep = ("learning_rate", "num_train_epochs", "per_device_train_batch_size",
            "gradient_accumulation_steps", "weight_decay", "warmup_ratio",
            "warmup_steps", "max_grad_norm", "label_smoothing_factor",
            "lr_scheduler_type", "fp16", "bf16", "optim", "seed")
    return {k: d[k] for k in keep if k in d}


# ============================================================================
# 6. 한 시드 학습
# ============================================================================
def run_seed(a, seed, tok, train_recs, test_recs, gen_recs, hp):
    set_seed(seed)
    tr, dv, dev_docs = split_dev_by_doc(train_recs, a.n_dev_docs, seed)
    ds_tr, ds_dv = JudgeDS(tr, tok, a), JudgeDS(dv, tok, a)
    print(f"\n[seed {seed}] train {len(tr)} / dev {len(dv)} claims   dev덱={dev_docs}")
    print(f"[seed {seed}] 입력 길이 p50/p95/max = "
          f"{int(np.percentile(ds_tr.lens,50))}/{int(np.percentile(ds_tr.lens,95))}/{max(ds_tr.lens)}"
          f"  (상한 {a.max_len})")

    model = AutoModelForSequenceClassification.from_pretrained(
        a.backbone, num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)}, label2id=L2I,
        ignore_mismatched_sizes=True)

    tmp = tempfile.mkdtemp(prefix=f"{a.model_id}_s{seed}_")
    targs = make_targs(
        output_dir=tmp, seed=seed,
        learning_rate=hp["learning_rate"], num_train_epochs=hp["num_train_epochs"],
        per_device_train_batch_size=hp["per_device_train_batch_size"],
        per_device_eval_batch_size=a.eval_batch_size,
        gradient_accumulation_steps=hp["gradient_accumulation_steps"],
        weight_decay=hp["weight_decay"], warmup_ratio=hp["warmup_ratio"],
        max_grad_norm=hp["max_grad_norm"],
        label_smoothing_factor=hp["label_smoothing_factor"],
        lr_scheduler_type=hp["lr_scheduler_type"],
        fp16=hp["fp16"] and torch.cuda.is_available(),
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, logging_steps=25, report_to=[], disable_tqdm=False)

    trainer = make_trainer(
        WeightedTrainer, model=model, args=targs,
        train_dataset=ds_tr, eval_dataset=ds_dv,
        tokenizer=tok, compute_metrics=compute_metrics,
        data_collator=PadCollator(tok.pad_token_id),
        class_weights=class_weights(tr, a.cw_power))
    trainer.train()

    out = {"seed": seed, "dev_docs": dev_docs,
           "dev_macro_f1": float(trainer.evaluate()["eval_macro_f1"])}

    for name, recs in (("test", test_recs), ("generalization", gen_recs)):
        if not recs:
            continue
        ds = JudgeDS(recs, tok, a)
        pred = trainer.predict(ds)
        logits = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        yp = list(np.argmax(logits, -1))
        yt = [r["labels"] for r in ds.rows]
        m = metrics_from(yt, yp)
        m["ci95"] = doc_bootstrap_ci(yt, yp, ds.docs, a.n_boot, seed)
        out[name] = m
        print(f"[seed {seed}] {name.upper():16s} macro-F1 {m['macro_f1']:.4f} "
              f"acc {m['accuracy']:.4f}  CI95 {m['ci95'][0]:.3f}~{m['ci95'][1]:.3f}")

    return out, trainer, tmp


# ============================================================================
# 7. README 자동 생성 — 중간발표 Q&A("파라미터 뭘로 했냐")에 그대로 답이 되도록
# ============================================================================
def write_readme(a, hp, summary, outdir):
    s, runs = summary, summary["runs"]
    best = max(runs, key=lambda r: r["dev_macro_f1"])
    cm = [[0] * 4 for _ in range(4)]
    for r in runs:
        if "test" in r:
            for i in range(4):
                for j in range(4):
                    cm[i][j] += r["test"]["confusion"][i][j]

    L = [f"# {a.model_id} — 판정 모델", "",
         f"- **구성**: {'mDeBERTa' if 'deberta' in a.backbone.lower() else a.backbone} 파인튜닝, "
         f"맥락 {'**포함**' if a.use_context else '없음'} "
         f"(claim {'+ 슬라이드 맥락 ' if a.use_context else ''}+ 후보 구간 top-{a.n_cand})",
         f"- **백본**: `{a.backbone}`",
         f"- **확인하려는 것**: "
         + ("슬라이드 맥락의 효과 (← 프로젝트 핵심 주장)" if a.use_context
            else "맥락 없는 대조군. M3 와 이 모델의 차이가 곧 맥락 효과"),
         f"- **git commit**: `{s.get('git_commit')}`", ""]

    L += ["## 왜 이 백본인가", "",
          "계획서의 `mDeBERTa-v3-base-mnli-xnli` 대신 M2 담당자가 채택한 백본과 동일하게 맞췄다. "
          "M2 에서 mDeBERTa 는 fp16 학습 중 loss=nan 이 재현되어 사용할 수 없었고, "
          "**M2' 와 M3 가 백본까지 동일해야 맥락 ablation 이 성립**하기 때문이다.", ""] \
        if "deberta" not in a.backbone.lower() else []

    L += ["## 입력 포맷", "", "```",
          "[CLS] claim [SEP] " + ("슬라이드 제목 + 같은 슬라이드 다른 불릿 [SEP] " if a.use_context else "")
          + f"후보 구간 1..{a.n_cand} [SEP]", "```", "",
          f"### 토큰 예산 (상한 {a.max_len})", "",
          "| 블록 | 상한 | 비고 |", "|---|---|---|",
          f"| claim | {a.claim_max} | 원자 명제 1개 |"]
    if a.use_context:
        L.append(f"| 슬라이드 맥락 | {a.ctx_max} | 제목 + 불릿 {a.ctx_bullets}개까지 |")
    L += [f"| 후보 × {a.n_cand} | {a.cand_max} × {a.n_cand} = {a.cand_max*a.n_cand} | 3문장 윈도우를 앞 {a.cand_max}토큰으로 절단 |",
          f"| 특수 토큰 | {2 + (1 if a.use_context else 0) + a.n_cand} | |", "",
          "블록마다 따로 자른다. HF 기본 `truncation='longest_first'` 에 맡기면 어느 블록이 "
          "희생되는지 통제할 수 없어 claim 이 밀려 나갈 수 있다.", "",
          f"> `{a.backbone}` 은 절대 위치 임베딩을 쓰므로 상한을 넘기면 런타임 에러가 난다. "
          f"실사용 상한이 512 가 아니라 510~511 이라 `max_len={a.max_len}` 로 고정했다.", ""]

    L += ["## 하이퍼파라미터", "", "| 항목 | 값 |", "|---|---|"]
    for k, v in [("learning_rate", hp["learning_rate"]),
                 ("num_train_epochs", hp["num_train_epochs"]),
                 ("per_device_train_batch_size", hp["per_device_train_batch_size"]),
                 ("gradient_accumulation_steps", hp["gradient_accumulation_steps"]),
                 ("유효 배치", hp["per_device_train_batch_size"] * hp["gradient_accumulation_steps"]),
                 ("weight_decay", hp["weight_decay"]), ("warmup_ratio", hp["warmup_ratio"]),
                 ("max_grad_norm", hp["max_grad_norm"]),
                 ("label_smoothing_factor", hp["label_smoothing_factor"]),
                 ("lr_scheduler_type", hp["lr_scheduler_type"]), ("fp16", hp["fp16"]),
                 ("클래스 가중치", f"역빈도^{a.cw_power} (근거있음 편중 보정)"),
                 ("early stopping", "dev macro-F1 기준, epoch 단위, best 복원"),
                 ("dev 분할", f"train 에서 **문서 단위** {a.n_dev_docs}덱 (시드마다 다름)"),
                 ("시드", a.seeds)]:
        L.append(f"| `{k}` | {v} |")
    if a.match_args:
        L += ["", f"위 값은 `{a.match_args}` 에서 그대로 읽어왔다 — M2 와 조건을 맞추기 위함."]
    L.append("")

    L += ["## 데이터", "", "| 파일 | sha256(12) | 건수 |", "|---|---|---|"]
    for k, v in (s.get("data_fingerprint") or {}).items():
        L.append(f"| `{k}` | `{v['sha12']}` | {v['lines']} |")
    L += ["", f"라벨 분포 (train): {s['train_label_dist']}", "",
          "> 해시를 기록하는 이유: `dataset/` 이 8/19~8/21 계속 갱신되었다. "
          "M2' 와 M3 가 같은 스냅샷으로 학습되었는지는 이 해시로만 증명된다.", ""]

    L += ["## 결과 (test)", "",
          "| 시드 | dev macro-F1 | test macro-F1 | accuracy | CI95 (문서 부트스트랩) |",
          "|---|---|---|---|---|"]
    for r in runs:
        t = r.get("test")
        L.append(f"| {r['seed']} | {r['dev_macro_f1']:.4f} | "
                 + (f"{t['macro_f1']:.4f} | {t['accuracy']:.4f} | "
                    f"{t['ci95'][0]:.3f}~{t['ci95'][1]:.3f} |" if t else " — | — | — |"))
    if "test_macro_f1_mean" in s:
        L += ["", f"**test macro-F1 = {s['test_macro_f1_mean']:.4f} ± "
                  f"{s['test_macro_f1_std']:.4f}** ({len(runs)} 시드)"]
    if "generalization_macro_f1_mean" in s:
        L.append(f"**generalization macro-F1 = {s['generalization_macro_f1_mean']:.4f} ± "
                 f"{s['generalization_macro_f1_std']:.4f}** (미지 분야 = 재무)")
    L += ["", "주 지표는 accuracy 가 아니라 **macro-F1** 이다. 근거있음이 다수 클래스라 "
              "accuracy 로 보고하면 \"전부 근거있음으로 찍어도 그 정도\"라는 지적이 즉시 들어온다.", ""]

    L += ["### 혼동행렬 (전체 시드 합산, 행=정답 / 열=예측)", "",
          "| gold \\ pred | " + " | ".join(LABELS) + " |", "|---|" + "---|" * 4]
    for i, lab in enumerate(LABELS):
        L.append(f"| **{lab}** | " + " | ".join(str(v) for v in cm[i]) + " |")
    L += ["", "## 저장된 모델", "",
          f"- 이 폴더의 가중치는 **dev macro-F1 최고 시드(seed {best['seed']}, "
          f"{best['dev_macro_f1']:.4f})** 의 것이다.",
          "- `model.safetensors` 는 용량 때문에 git 에 올리지 않는다 (`.gitignore`). "
          "Drive / 로컬에 보관하고 필요하면 아래 명령으로 재현한다.", "",
          "## 재현", "", "```bash",
          f"python src/train_judge_m3.py --model_id {a.model_id} "
          + ("--use_context " if a.use_context else "")
          + f"\\\n       --backbone {a.backbone} --max_len {a.max_len}"
          + (f" \\\n       --match_args {a.match_args}" if a.match_args else ""),
          "```", "",
          "## 관련 산출물", "",
          f"- `results/{a.model_id}_test.json` — 시드별 원시 결과",
          "- `results/model_comparison.md` — M1 / M2 / M2' / M3 비교표",
          "- `src/check_ablation.py` — M2' 와 M3 가 정말 `--use_context` 하나만 다른지 기계 검증", ""]

    (outdir / f"{a.model_id}_README.md").write_text("\n".join(L), encoding="utf-8")


# ============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="M3")
    p.add_argument("--use_context", action="store_true",
                   help="★ M3 와 M2' 를 가르는 유일한 플래그")
    p.add_argument("--backbone", default="klue/roberta-base")
    p.add_argument("--data_dir", default="dataset")
    p.add_argument("--out_root", default="models")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--match_args", default=None,
                   help="시나님 models/M2/training_args.bin — 하이퍼파라미터를 그대로 채택")
    p.add_argument("--eval_generalization", action="store_true",
                   help="Phase D 전에는 켜지 말 것")

    # 토큰 예산
    p.add_argument("--max_len", type=int, default=510)
    p.add_argument("--claim_max", type=int, default=64)
    p.add_argument("--ctx_max", type=int, default=96)
    p.add_argument("--ctx_bullets", type=int, default=6)
    p.add_argument("--n_cand", type=int, default=5)
    p.add_argument("--cand_max", type=int, default=64)

    # 학습 (match_args 가 있으면 그쪽이 이긴다)
    p.add_argument("--seeds", default="13,42,777")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=float, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--scheduler", default="linear")
    p.add_argument("--cw_power", type=float, default=0.5)
    p.add_argument("--n_dev_docs", type=int, default=3)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--no_fp16", dest="fp16", action="store_false", default=True)
    p.add_argument("--inspect", action="store_true",
                   help="학습하지 않고, 조립된 입력을 디코딩해서 보여주고 끝낸다")
    a = p.parse_args()

    hp = {"learning_rate": a.lr, "num_train_epochs": a.epochs,
          "per_device_train_batch_size": a.batch_size,
          "gradient_accumulation_steps": a.grad_accum, "weight_decay": a.weight_decay,
          "warmup_ratio": a.warmup_ratio, "max_grad_norm": a.max_grad_norm,
          "label_smoothing_factor": a.label_smoothing,
          "lr_scheduler_type": a.scheduler, "fp16": a.fp16}
    if a.match_args:
        got = read_match_args(a.match_args)
        print(f"[match_args] {a.match_args} 에서 채택: "
              f"{ {k: v for k, v in got.items() if k in hp} }")
        hp.update({k: v for k, v in got.items() if k in hp})

    d = Path(a.data_dir)
    print(f"\n{'='*70}\n {a.model_id}   backbone={a.backbone}   use_context={a.use_context}\n{'='*70}")

    print("[train.jsonl]")
    train_recs = normalize(load_jsonl(d / "train.jsonl"))
    test_recs, gen_recs = [], []
    if (d / "test.jsonl").exists():
        print("[test.jsonl]")
        test_recs = normalize(load_jsonl(d / "test.jsonl"))
    if a.eval_generalization and (d / "generalization.jsonl").exists():
        print("[generalization.jsonl]")
        gen_recs = normalize(load_jsonl(d / "generalization.jsonl"))
    if not train_recs:
        raise SystemExit("\n학습 가능한 행이 0개다. `python src/inspect_dataset.py` 로 원인을 확인할 것.")

    dist = dict(Counter(r["label"] for r in train_recs))
    print(f"train {len(train_recs)} / test {len(test_recs)} / gen {len(gen_recs)} claims")
    print(f"train 라벨 분포: {dist}")

    tok = AutoTokenizer.from_pretrained(a.backbone)

    if a.inspect:
        ds = JudgeDS(train_recs, tok, a)
        import numpy as _np
        print(f"\n입력 길이 p50/p90/p95/max = "
              f"{int(_np.percentile(ds.lens,50))}/{int(_np.percentile(ds.lens,90))}/"
              f"{int(_np.percentile(ds.lens,95))}/{max(ds.lens)}   (상한 {a.max_len})")
        over = sum(1 for x in ds.lens if x >= a.max_len)
        print(f"상한에 닿아 잘린 행: {over}/{len(ds.lens)} ({over/len(ds.lens):.1%})")
        print(f"\n--- 실제로 모델에 들어가는 시퀀스 (첫 행) ---\n")
        print(tok.decode(ds.rows[0]["input_ids"]))
        print(f"\n라벨: {LABELS[ds.rows[0]['labels']]}")
        print("\n★ claim 이 맨 앞에 온전히 보이는지 확인할 것. 잘려 있으면 --claim_max 를 올린다.")
        return

    runs, best_dev, best_trainer, tmps = [], -1.0, None, []
    for s in a.seeds.split(","):
        r, trainer, tmp = run_seed(a, int(s), tok, train_recs, test_recs, gen_recs, hp)
        runs.append(r)
        tmps.append(tmp)
        if r["dev_macro_f1"] > best_dev:
            best_dev, best_trainer = r["dev_macro_f1"], trainer

    summary = {"model_id": a.model_id, "use_context": a.use_context, "backbone": a.backbone,
               "config": vars(a), "hyperparameters": hp, "runs": runs,
               "train_label_dist": dist, "data_fingerprint": fingerprint(d),
               "git_commit": git_commit()}
    for split in ("test", "generalization"):
        v = [r[split]["macro_f1"] for r in runs if split in r]
        if v:
            summary[f"{split}_macro_f1_mean"] = float(np.mean(v))
            summary[f"{split}_macro_f1_std"] = float(np.std(v))

    # ---- 시나님 M2 와 동일한 폴더 구성으로 저장 ----
    outdir = Path(a.out_root) / a.model_id
    outdir.mkdir(parents=True, exist_ok=True)
    best_trainer.save_model(str(outdir))      # config.json + model.safetensors + training_args.bin
    tok.save_pretrained(str(outdir))          # tokenizer.json + tokenizer_config.json + ...
    write_readme(a, hp, summary, outdir)

    Path(a.results_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.results_dir) / f"{a.model_id}_test.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    for t in tmps:
        shutil.rmtree(t, ignore_errors=True)

    print(f"\n{'='*70}")
    print(f"저장 완료: {outdir}/")
    for f in sorted(p.name for p in outdir.iterdir()):
        print(f"   {f}")
    print(f"결과: {a.results_dir}/{a.model_id}_test.json")
    if "test_macro_f1_mean" in summary:
        print(f"\ntest macro-F1 = {summary['test_macro_f1_mean']:.4f} "
              f"± {summary['test_macro_f1_std']:.4f} ({len(runs)} 시드)")
    print("\n다음: M2' 도 돌리고 (--use_context 만 빼면 된다) 아래로 검증")
    print(f"  python src/check_ablation.py --base results/M2p_test.json "
          f"--variant results/{a.model_id}_test.json")


if __name__ == "__main__":
    main()
