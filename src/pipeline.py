import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import os
import torch
import numpy as np
from typing import List, Dict, Any
from pathlib import Path

# 기존 모듈 임포트
from segment import split_sentences, build_spans
from claim_split import read_slides, build_records
from retrieve_bm25 import BM25Index, retrieve as retrieve_bm25, DEFAULT_K1, DEFAULT_B
from dense_retrieve import load_model as load_dense_model, encode as encode_dense, DEFAULT_MODEL as DENSE_DEFAULT_MODEL

class FactCheckerPipeline:
    def __init__(self, m3_model_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. 임베딩 모델 로드 (검색용)
        self.embed_model = load_dense_model(DENSE_DEFAULT_MODEL, self.device)
        
        # 2. 판정 모델 로드 (M3)
        self.label_list = ["근거있음", "무근거", "모순", "benign"]
        self.judge_model = None
        self.judge_tokenizer = None
        
        if m3_model_path and os.path.exists(m3_model_path):
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.judge_tokenizer = AutoTokenizer.from_pretrained(m3_model_path)
            self.judge_model = AutoModelForSequenceClassification.from_pretrained(m3_model_path).to(self.device)
            self.judge_model.eval()

    def _hybrid_search(self, claims: List[Dict], spans: List[Dict], top_n: int = 5) -> Dict[str, List[Dict]]:
        """BM25와 Dense(e5)를 RRF 방식으로 융합하여 Top-N 반환"""
        if not claims or not spans:
            return {}

        temp_queries = [
            {
                "claim_id": str(i), 
                "query_text": c["Claim (PPT)"],
                "deck_id": "temp_deck",
                "doc_id": "temp_doc"
            } 
            for i, c in enumerate(claims)
        ]
        formatted_spans = [{"passage_id": s.span_id, "text": s.text, "sent_ids": s.sent_ids, "window": "w3"} for s in spans]
        
        # BM25 검색
        bm25_idx = BM25Index(formatted_spans, k1=DEFAULT_K1, b=DEFAULT_B)
        bm25_res = retrieve_bm25(temp_queries, formatted_spans, bm25_idx, top_n=60)
        
        # Dense 검색
        p_texts = [s["text"] for s in formatted_spans]
        p_vecs = encode_dense(self.embed_model, p_texts, "passage: ")
        q_texts = [q["query_text"] for q in temp_queries]
        q_vecs = encode_dense(self.embed_model, q_texts, "query: ")
        
        sims = q_vecs @ p_vecs.T
        dense_res = []
        for i, q in enumerate(temp_queries):
            dense_top_idx = np.argsort(-sims[i])[:60]
            results = [{"passage_id": formatted_spans[idx]["passage_id"], "rank": r} for r, idx in enumerate(dense_top_idx, start=1)]
            dense_res.append({"claim_id": q["claim_id"], "results": results})

        # RRF 융합 (k=60)
        k = 60
        hybrid_candidates = {}
        span_lookup = {s["passage_id"]: s for s in formatted_spans}
        
        for q_idx, q in enumerate(temp_queries):
            cid = q["claim_id"]
            rrf_score = {}
            for hit in bm25_res[q_idx]["results"] + dense_res[q_idx]["results"]:
                pid = hit["passage_id"]
                rrf_score[pid] = rrf_score.get(pid, 0.0) + (0.5 / (k + hit["rank"]))
                
            sorted_pids = sorted(rrf_score.keys(), key=lambda x: -rrf_score[x])[:top_n]
            hybrid_candidates[cid] = [span_lookup[pid] for pid in sorted_pids]
            
        return hybrid_candidates

    def _encode_m3_input(self, claim_text: str, slide_title: str, slide_context: List[str], candidates: List[Dict]) -> Dict[str, torch.Tensor]:
        """M3 학습 코드의 토큰 예산 규칙에 맞춘 정밀 절단 로직 (max_len=510)"""
        tok = self.judge_tokenizer
        cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
        
        def enc(text, cap):
            if not text: return []
            return tok(str(text), add_special_tokens=False)["input_ids"][:cap]

        # 1. Claim (최대 64토큰)
        ids = [cls_id] + enc(claim_text, 64) + [sep_id]

        # 2. Context (최대 96토큰, 불릿 6개)
        ctx_parts = [slide_title] + slide_context[:6]
        ids += enc(" / ".join(p for p in ctx_parts if p), 96) + [sep_id]

        # 3. Candidates (각 64토큰씩 최대 5개)
        for c in candidates[:5]:
            cids = enc(c["text"], 64)
            if cids:
                ids += cids + [sep_id]

        ids = ids[:510]
        
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long).to(self.device),
            "attention_mask": torch.tensor([[1] * len(ids)], dtype=torch.long).to(self.device)
        }

    def check(self, doc_path: str, deck_path: str) -> Dict[str, Any]:
        """
        입력: 문서 경로, PPT 경로
        출력: claim별 판정, 모델이 선별한 근거 구간 (Top-1 적용)
        """
        raw_text = Path(doc_path).read_text(encoding="utf-8")
        sents = split_sentences(raw_text)
        w3_spans = build_spans(sents, 3)
        
        slides = read_slides(Path(deck_path))
        claims = [row for slide in slides for row in build_records(slide)]
        
        candidates_map = self._hybrid_search(claims, w3_spans, top_n=5)
        
        results = []
        for i, claim in enumerate(claims):
            cid = str(i)
            cands = candidates_map.get(cid, [])
            
            pred_label = "판정 대기"
            selected_evidence = "판정 대기 (모델 미연결)"
            
            if self.judge_model:
                # M3 전용 포맷터로 입력 생성
                inputs = self._encode_m3_input(
                    claim["Claim (PPT)"], claim["Slide_Title"], claim["Context (PPT)"], cands
                )
                    
                with torch.no_grad():
                    logits = self.judge_model(**inputs).logits[0]
                    pred_idx = torch.argmax(logits).item()
                    pred_label = self.label_list[pred_idx]
                
                # 라벨에 따른 근거 선별 로직 (1등 근거 활용)
                if pred_label in ["근거있음", "모순"] and cands:
                    selected_evidence = cands[0]["text"]  # 가장 점수 높은 1등 문장
                elif pred_label in ["무근거", "benign"]:
                    selected_evidence = "해당 없음 (원문에 근거 없음)"
                else:
                    selected_evidence = "근거 후보 없음"
                
            results.append({
                "slide_number": claim["Slide #"],
                "claim": claim["Claim (PPT)"],
                "predicted_label": pred_label,
                "selected_evidence": selected_evidence,
                "top5_candidates": [{"passage_id": c["passage_id"], "text": c["text"]} for c in cands]
            })
            
        return {
            "document": doc_path,
            "presentation": deck_path,
            "results": results
        }


#################################
# 테스트 코드. 지워도 됩니다 
###################################

if __name__ == "__main__":
    import json
    import sys
    
    # 윈도우 환경 OpenMP 충돌 방지
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    test_doc = "cat.txt"  
    test_deck = "cat.pptx"
    
    if not Path(test_doc).exists() or not Path(test_deck).exists():
        print("테스트 파일이 없습니다. 경로를 수정하거나 파일을 준비해 주세요.")
        sys.exit(1)

    print(" 파이프라인 초기화 중...")
    pipeline = FactCheckerPipeline(m3_model_path=None) # 모델 경로가 생기면 여기에 "models/M3"를 넣으세요
    
    print(f"\n [{test_deck}] 문서를 바탕으로 팩트체크 시작...")
    result = pipeline.check(test_doc, test_deck)
        
    print("\n 파이프라인 실행 성공! 최종 출력(JSON) 결과:")
    print(json.dumps(result, ensure_ascii=False, indent=2))