import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 모듈 임포트
try:
    from src.segment import split_sentences, build_spans, _write_jsonl
    from src.claim_split import read_slides, build_records, save_jsonl
    from src.retrieve_bm25 import BM25Index, retrieve as retrieve_bm25, write_jsonl as write_bm25_jsonl
    from src.dense_retrieve import run_deck as run_dense_deck, DEFAULT_MODEL as DENSE_DEFAULT_MODEL
    from src.build_dataset import build_dataset
except ImportError:
    from segment import split_sentences, build_spans, _write_jsonl
    from claim_split import read_slides, build_records, save_jsonl
    from retrieve_bm25 import BM25Index, retrieve as retrieve_bm25, write_jsonl as write_bm25_jsonl
    try:
        from dense_retrieve import run_deck as run_dense_deck, DEFAULT_MODEL as DENSE_DEFAULT_MODEL
    except ImportError:
        run_dense_deck = None
        DENSE_DEFAULT_MODEL = "intfloat/multilingual-e5-large"
    from build_dataset import build_dataset


def run_full_pipeline(doc_id: str, deck_id: str, root: Path = Path(".")) -> bool:
    print(f"[Pipeline] Starting verification for doc_id={doc_id}, deck_id={deck_id}")

    # Step 1: A-2 문장 분할 및 구간 생성
    print("[Step 1] Running A-2 segmentation...")
    clean_txt_path = root / "docs" / "clean" / f"{doc_id}.txt"
    if not clean_txt_path.exists():
        clean_txt_path = root / "docs" / f"{doc_id}.txt"

    if not clean_txt_path.exists():
        txt_candidates = list((root / "docs").glob(f"*{doc_id}*.txt"))
        if txt_candidates:
            clean_txt_path = txt_candidates[0]
        else:
            print(f"[Error] Source document not found: docs/{doc_id}.txt", file=sys.stderr)
            return False

    raw_text = clean_txt_path.read_text(encoding="utf-8")
    sents = split_sentences(raw_text)
    w3_spans = build_spans(sents, 3)

    out_w3_path = root / "spans" / "w3" / f"{doc_id}.jsonl"
    _write_jsonl(root / "spans" / "sents" / f"{doc_id}.jsonl", sents)
    _write_jsonl(out_w3_path, w3_spans)

    passages = [{"passage_id": sp.span_id, "text": sp.text, "sent_ids": sp.sent_ids, "window": "w3"} for sp in w3_spans]
    passages_out = root / "passages" / f"{doc_id}.jsonl"
    passages_out.parent.mkdir(parents=True, exist_ok=True)
    with passages_out.open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"[Step 1] Completed: {len(w3_spans)} spans generated.")

    # Step 2: A-3 PPT Claim 분할
    print("[Step 2] Running A-3 claim extraction...")
    pptx_path = root / "decks" / f"{deck_id}.pptx"
    if not pptx_path.exists():
        candidates = list((root / "decks").glob(f"*{doc_id}*.pptx"))
        if candidates:
            pptx_path = candidates[0]

    claims_rows = []
    if pptx_path.exists():
        slides = read_slides(pptx_path)
        claims_rows = [row for slide in slides for row in build_records(slide)]
        out_claim_path = root / "claims" / f"{deck_id}.jsonl"
        save_jsonl(out_claim_path, claims_rows)
        print(f"[Step 2] Completed: {len(claims_rows)} claims extracted from PPTX.")
    else:
        fallback_claim_path = root / "claims" / f"{deck_id}.jsonl"
        if fallback_claim_path.exists():
            with fallback_claim_path.open(encoding="utf-8") as f:
                claims_rows = [json.loads(line) for line in f if line.strip()]
            print(f"[Step 2] Loaded {len(claims_rows)} claims from existing JSONL.")
        else:
            print(f"[Error] PPTX or Claim JSONL not found.", file=sys.stderr)
            return False

    queries = []
    for idx, c in enumerate(claims_rows, start=1):
        c_id = c.get("claim_id") or f"{deck_id}__s{c.get('Slide #', 1)}__c{idx}"
        q_text = c.get("Claim (PPT)") or c.get("claim_text") or ""
        queries.append({
            "claim_id": c_id,
            "deck_id": deck_id,
            "doc_id": doc_id,
            "query_text": q_text
        })
    query_out = root / "retrieval" / "queries" / f"{deck_id}.jsonl"
    query_out.parent.mkdir(parents=True, exist_ok=True)
    with query_out.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Step 3: A-4 BM25 키워드 검색
    print("[Step 3] Running A-4 BM25 retrieval...")
    index = BM25Index(passages)
    bm25_results = retrieve_bm25(queries, passages, index, top_n=5)
    out_bm25_path = root / "retrieval" / "bm25" / f"{deck_id}.jsonl"
    write_bm25_jsonl(out_bm25_path, bm25_results)
    print(f"[Step 3] Completed: BM25 results saved to {out_bm25_path}")

    # Step 4: A-5 Dense 임베딩 검색
    print("[Step 4] Running A-5 Dense retrieval...")
    if run_dense_deck is not None:
        try:
            status = run_dense_deck(
                root=root,
                deck_id=deck_id,
                model_name=DENSE_DEFAULT_MODEL,
                device=None,
                top_n=5,
                window="w3",
                show=0
            )
            if status == 0:
                print(f"[Step 4] Completed: Dense results saved to retrieval/dense/{deck_id}.jsonl")
            else:
                print(f"[Step 4] Dense retrieval returned code {status}")
        except Exception as e:
            print(f"[Step 4] Dense retrieval execution skipped or failed: {e}")
    else:
        print("[Step 4] dense_retrieve module not found. Skipped.")

    # Step 5: B-1 Hybrid 검색기 확인
    print("[Step 5] Checking B-1 Hybrid retrieval...")
    hybrid_candidates = [
        root / "src" / "retrieve_hybrid.py",
        root / "src" / "hybrid_retrieve.py",
        root / "src" / "fuse_ranks.py"
    ]
    b1_executed = False
    for h_script in hybrid_candidates:
        if h_script.exists():
            try:
                cmd = [sys.executable, str(h_script), "--deck-id", deck_id]
                subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(f"[Step 5] Completed: B-1 script ({h_script.name}) executed.")
                b1_executed = True
                break
            except Exception as e:
                print(f"[Step 5] B-1 execution failed: {e}")
    if not b1_executed:
        print("[Step 5] B-1 hybrid script not found yet. Skipped.")

    # Step 6: B-2 데이터셋 생성
    print("[Step 6] Running B-2 dataset builder...")
    excel_path = root / "annotation.xlsx"
    manifest_path = root / "docs" / "manifest.csv"
    build_dataset(
        root, 
        manifest_path, 
        root / "dataset", 
        target_deck=deck_id, 
        excel_file=excel_path if excel_path.exists() else None
    )
    print(f"[Step 6] Completed: dataset/*.jsonl updated.")

    print(f"[Pipeline] Full End-to-End execution finished for {deck_id}.")
    return True


def main():
    parser = argparse.ArgumentParser(description="End-to-End pipeline verification script.")
    parser.add_argument("--doc-id", default="arts_01")
    parser.add_argument("--deck-id", default="arts_01__claudecode")
    parser.add_argument("--root", type=Path, default=ROOT_DIR)
    args = parser.parse_args()

    success = run_full_pipeline(args.doc_id, args.deck_id, args.root)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()