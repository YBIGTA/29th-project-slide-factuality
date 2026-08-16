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
    from src.build_passages import build as build_passages, write_jsonl as write_passages
    from src.build_queries import build as build_queries, write_jsonl as write_queries, load_manifest
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
    from build_passages import build as build_passages, write_jsonl as write_passages
    from build_queries import build as build_queries, write_jsonl as write_queries, load_manifest


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

    # passage 는 A-5 의 build_passages 를 그대로 쓴다. 여기서 따로 만들면
    # passage_id 에 doc_id 접두사가 빠져서(`w3_001` vs `arts_01_w3_001`)
    # 문서 간 ID 가 겹치고, 검색 결과에서 doc_id 를 떼어 정답과 맞추는
    # B-1 의 대조가 통째로 깨진다 (README_A5 "팀 계약").
    passages = build_passages(root, doc_id, "w3")
    write_passages(root / "passages" / f"{doc_id}.jsonl", passages)
    print(f"[Step 1] Completed: {len(w3_spans)} spans generated, "
          f"{len(passages)} passages ({passages[0]['passage_id']} ...).")

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

    # 쿼리도 A-5 의 build_queries 를 그대로 쓴다. 여기서 따로 만들면
    # claim_id 가 옛 규약(`{deck}__s1__c1`)으로 붙어서, build_queries 가
    # 만드는 `{deck}_s01_c01` 과 안 맞고 dataset↔retrieval 조인이 0건이 된다.
    queries = build_queries(root, deck_id, load_manifest(root))
    write_queries(root / "retrieval" / "queries" / f"{deck_id}.jsonl", queries)
    print(f"[Step 2] claim_id 부여: {queries[0]['claim_id']} ...")

    # Step 3: A-4 BM25 키워드 검색
    print("[Step 3] Running A-4 BM25 retrieval...")
    index = BM25Index(passages)
    # top_n 은 20 으로 맞춘다. 이 스크립트는 실제 retrieval/ 파일을 덮어쓰므로
    # 5 로 돌리면 R@10 을 못 재게 되어 B-1 지표가 반쪽이 된다.
    bm25_results = retrieve_bm25(queries, passages, index, top_n=20)
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
                top_n=20,
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

    # Step 5: B-1 하이브리드 + 검색 방식 비교
    # 하이브리드는 별도 검색기가 아니라 BM25·임베딩 결과를 RRF 로 합치는
    # 계산이라, compare_retrieval.py 안에서 만들어진다.
    print("[Step 5] Running B-1 retrieval comparison...")
    b1 = root / "src" / "compare_retrieval.py"
    ann = root / "annotation" / "annotations_long.csv"
    if not b1.exists():
        print("[Step 5] compare_retrieval.py 가 없다. 건너뜀.")
    elif not ann.exists():
        print(f"[Step 5] 라벨이 없다 ({ann}). 하이브리드 평가는 라벨이 있어야 한다. 건너뜀.")
    else:
        cmd = [sys.executable, str(b1), "--root", str(root), "--deck-id", deck_id,
               "--annotations", str(ann),
               "--out", str(root / "results" / f"retrieval_compare_{deck_id}.md"),
               "--metrics-dir", str(root / "results" / "_metrics")]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if r.returncode == 0:
            print(f"[Step 5] Completed: results/retrieval_compare_{deck_id}.md")
        else:
            print(f"[Step 5] 실패 (code {r.returncode})\n{(r.stderr or '')[-400:]}")

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