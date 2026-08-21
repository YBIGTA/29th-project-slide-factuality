#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
src/inspect_dataset.py — 학습 전에 dataset/*.jsonl 이 쓸 수 있는 상태인지 진단한다.

학습 25분 돌리고 죽는 대신 10초로 끝낸다. 출력 전체를 그대로 공유하면 된다.

python src/inspect_dataset.py
python src/inspect_dataset.py --data_dir dataset
"""
import argparse
import json
from collections import Counter
from pathlib import Path

# 같은 뜻으로 쓰인 여러 키 이름. 레포에 계획서 스키마와 은채님 스프레드시트 형식이
# 섞여 있어서 양쪽을 다 받는다.
ALIAS = {
    "claim_text":    ["claim_text", "Claim (PPT)", "claim", "Claim", "주장 텍스트", "text"],
    "label":         ["label", "Label", "라벨", "gold", "판정"],
    "doc_id":        ["doc_id", "docid", "document_id"],
    "deck_id":       ["deck_id", "deckid"],
    "claim_id":      ["claim_id", "claimid", "id"],
    "slide_title":   ["slide_title", "Slide_Title", "Slide Title", "슬라이드 제목", "슬라이드명"],
    "slide_context": ["slide_context", "Context (PPT)", "context", "Context", "맥락"],
    "slide_idx":     ["slide_idx", "Slide #", "slide_no", "슬라이드 넘버", "slide"],
    "candidates":    ["candidates", "cands", "top5", "retrieved", "candidate_spans"],
    "evidence_span": ["evidence_span", "evidence_span_id", "gold_span", "근거 구간"],
    "evidence_text": ["evidence_text", "Evidence Text", "증거 텍스트", "evidence"],
}
CAND_TEXT = ["text", "span_text", "passage", "sent", "구간", "content"]
CAND_ID = ["span_id", "id", "sid", "sent_id"]
LABELS = ["근거있음", "무근거", "모순", "benign"]


def pick(rec, canon):
    for k in ALIAS[canon]:
        if k in rec and rec[k] not in (None, ""):
            return k, rec[k]
    return None, None


def report(path):
    print(f"\n{'='*72}\n {path}\n{'='*72}")
    if not Path(path).exists():
        print("  ❌ 파일이 없다.")
        return
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    print(f"  건수: {len(recs)}")
    if not recs:
        return

    # 실제로 존재하는 키
    keys = Counter(k for r in recs for k in r)
    print(f"\n  --- 실제 키 (등장 횟수 / {len(recs)}) ---")
    for k, c in keys.most_common():
        print(f"    {k:<28} {c}")

    # 필요한 필드가 어떤 이름으로 들어있는지
    print("\n  --- 필드 매핑 ---")
    found = {}
    for canon in ALIAS:
        hits = Counter()
        for r in recs:
            k, _ = pick(r, canon)
            hits[k] += 1
        best = [k for k in hits if k][:1]
        ok = sum(v for k, v in hits.items() if k)
        found[canon] = best[0] if best else None
        mark = "✅" if ok == len(recs) else ("⚠ " if ok else "❌")
        src = best[0] if best else "(없음)"
        print(f"    {mark} {canon:<16} ← {src:<22} 채워진 행 {ok}/{len(recs)}")

    # 라벨 값
    lk = found.get("label")
    if lk:
        vals = Counter(str(r.get(lk, "")).strip() for r in recs)
        print(f"\n  --- 라벨 값 분포 ({lk}) ---")
        for v, c in vals.most_common():
            known = "✅" if v.lower() in [x.lower() for x in LABELS] or v in LABELS else "❌ 미인식"
            print(f"    {known} {v!r:<26} {c}")
    else:
        print("\n  ❌ 라벨 컬럼을 못 찾았다. 학습 불가.")

    # 후보 구간 — 판정 모델 입력의 핵심
    ck = found.get("candidates")
    print("\n  --- 후보 구간(검색기 top-5) ---")
    if not ck:
        print("    ❌ 없다. B-1/B-2 결과가 dataset/ 에 병합되지 않았다.")
        print("       이대로 학습하면 판정 모델이 원문 근거를 전혀 못 보고 claim 만으로 판정한다.")
    else:
        n = [len(r.get(ck) or []) for r in recs]
        print(f"    후보 개수 p50/최소/최대: {sorted(n)[len(n)//2]} / {min(n)} / {max(n)}")
        print(f"    후보 0개인 행: {sum(1 for x in n if x == 0)}")
        sample = next((r[ck][0] for r in recs if r.get(ck)), None)
        if sample:
            print(f"    후보 항목 키: {list(sample)}")
            tk = next((k for k in CAND_TEXT if k in sample), None)
            ik = next((k for k in CAND_ID if k in sample), None)
            print(f"    → 텍스트 키: {tk or '❌ 못 찾음'} / id 키: {ik or '❌ 못 찾음'}")
            if tk:
                print(f"    예시: {str(sample[tk])[:90]}...")

        # 정답 근거가 후보 안에 있는가 = 검색기가 씌운 성능 상한
        ek, ik2 = found.get("evidence_span"), None
        if ek and sample:
            ik2 = next((k for k in CAND_ID if k in sample), None)
        if ek and ik2:
            inside = sum(1 for r in recs
                         if r.get(ek) in [c.get(ik2) for c in (r.get(ck) or [])])
            print(f"\n    정답 근거가 후보 안에 있는 행: {inside}/{len(recs)} "
                  f"({inside/len(recs):.1%})")
            print("    → 이 비율이 판정 모델의 구조적 상한이다 (나머지는 원리적으로 맞출 수 없다)")

    # 문서 단위 분할 가능한가
    dk = found.get("doc_id") or found.get("deck_id")
    if dk:
        docs = Counter(r.get(dk) for r in recs)
        print(f"\n  --- 문서 ({dk}) ---")
        print(f"    문서 수: {len(docs)}   문서당 claim p50: {sorted(docs.values())[len(docs)//2]}")
        if len(docs) < 5:
            print(f"    ⚠ 문서가 {len(docs)}개뿐이라 문서 단위 dev 분할이 빡빡하다 "
                  f"(--n_dev_docs 를 낮출 것)")
    else:
        print("\n  ❌ doc_id / deck_id 가 없다. 문서 단위 분할이 불가능하고, "
              "claim 단위로 나누면 데이터 누수다.")

    # 통째로 못 쓰는 행
    bad = sum(1 for r in recs
              if not pick(r, "claim_text")[1] or not pick(r, "label")[1])
    print(f"\n  --- 학습 불가 행 (claim 또는 라벨 없음): {bad}/{len(recs)} ---")
    if bad:
        ex = next(r for r in recs
                  if not pick(r, "claim_text")[1] or not pick(r, "label")[1])
        print(f"    예시: {json.dumps(ex, ensure_ascii=False)[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="dataset")
    a = ap.parse_args()
    for n in ("train.jsonl", "test.jsonl", "generalization.jsonl"):
        report(Path(a.data_dir) / n)
    print(f"\n{'='*72}\n 이 출력 전체를 그대로 공유하면 된다.\n{'='*72}")


if __name__ == "__main__":
    main()
