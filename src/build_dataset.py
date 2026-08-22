from __future__ import annotations
"""
각자 맡으신 덱에 대해 검색기 실행과 라벨링이 완료되면 아래 명령어로 최종 `dataset/*.jsonl`을 생성해 주시면 됩니다.


`annotation.xlsx` 또는 `labels/` 폴더의 라벨과 `retrieval/` 검색 결과를 취합해 `docs/manifest.csv`의 split(train/test/generalization)에 맞춰 자동 분할 저장합니다.
```bash
python src/build_dataset.py --excel annotation.xlsx
"""
##############################

"""Phase B-2 · Input Data 가공 및 최종 데이터셋 빌더

- claims/, retrieval/ (dense/bm25), passages/ 와 결합하여 dataset/*.jsonl 생성
"""


import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


KNOWN_ANNOTATORS = ["시나", "윤서", "지원", "은채", "예린"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            with path.open(encoding=enc) as f:
                return [json.loads(line) for line in f if line.strip()]
        except Exception:
            continue
    return []


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    # 1. 기존 데이터가 있으면 모두 읽어오기
    existing_data = []
    if path.exists():
        existing_data = read_jsonl(path)
    
    # 2. 이번에 새로 갱신되는 덱 확인 — **doc_id 로 비교한다.**
    #    deck_id 로 비교하면 규약이 바뀌었을 때(`socio_04` -> `socio_04__claudecode`)
    #    옛 행이 '갱신 대상이 아니다' 로 잡혀 살아남고, 같은 claim 이 두 벌
    #    들어간다. 실제로 그렇게 test 가 290행에서 586행으로 불어난 적이 있다.
    def key(r):
        return str(r.get("doc_id") or r.get("deck_id", "")).split("__")[0]

    new_docs = {key(r) for r in rows if key(r)}

    # 3. 기존 데이터 중, 이번 갱신 대상이 '아닌' 데이터만 남기고 새 데이터와 결합
    merged_rows = [r for r in existing_data if key(r) not in new_docs]
    merged_rows.extend(rows)

    # 4. 결합된 전체 데이터를 저장
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in merged_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def load_manifest(manifest_path: Path) -> dict[str, dict[str, str]]:
    if not manifest_path.exists():
        return {}
    
    mapping = {}
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            with manifest_path.open(encoding=enc) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    deck_id = row.get("deck_id") or row.get("doc_id")
                    if deck_id:
                        mapping[deck_id.strip()] = {k.strip(): v.strip() for k, v in row.items() if k}
            if mapping:
                break
        except Exception:
            continue
    return mapping


def load_passages(root: Path, doc_id: str) -> dict[str, dict[str, Any]]:
    paths = [
        root / "passages" / f"{doc_id}.jsonl",
        root / "spans" / "w3" / f"{doc_id}.jsonl",
        root / "spans" / "sents" / f"{doc_id}.jsonl",
    ]
    for p in paths:
        if p.exists():
            rows = read_jsonl(p)
            return {r.get("passage_id") or r.get("span_id") or r.get("sent_id"): r for r in rows}
    return {}


_CLAIM_SUFFIX = re.compile(r"_(s\d+_c\d+)$")


def claim_suffix(claim_id: str) -> str | None:
    """`arts_01__claudecode_s03_c05` -> `s03_c05`.

    레포에 deck_id 표기가 두 가지 있다. manifest·dataset 은 짧은 `tech_02`,
    decks·claims·retrieval 은 파일명 그대로인 긴 `tech_02__claudecode` 다.
    그래서 같은 claim 인데 claim_id 가 `tech_02_s01_c01` 과
    `tech_02__claudecode_s01_c01` 로 갈린다. 슬라이드·순번 꼬리는 양쪽이
    같으므로 이걸 조인 키로 함께 쓴다.
    """
    m = _CLAIM_SUFFIX.search(claim_id or "")
    return m.group(1) if m else None


def load_retrieval_results(root: Path, deck_id: str) -> tuple[dict[str, list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    """검색 결과를 claim_id 로 찾을 수 있게 적재한다.

    파일명도 두 표기가 섞여 있어서(`tech_02.jsonl` vs `tech_02__claudecode.jsonl`)
    정확한 이름 다음에 `{doc_id}__*.jsonl` 글롭까지 훑는다. 이게 없으면 파일이
    멀쩡히 있는데도 못 찾아 candidates 가 조용히 빈 배열로 채워진다.
    """
    doc_id = deck_id.split("__")[0]
    exact = [
        root / "retrieval" / "hybrid_v3" / f"{deck_id}.jsonl",
        root / "retrieval" / "dense" / f"{deck_id}.jsonl",
        root / "retrieval" / "bm25" / f"{deck_id}.jsonl",
        root / "retrieval" / "dense" / f"{doc_id}.jsonl",
        root / "retrieval" / "bm25" / f"{doc_id}.jsonl",
    ]
    globbed: list[Path] = []
    for sub in ("hybrid_v3", "dense", "bm25"):
        d = root / "retrieval" / sub
        if d.is_dir():
            # `.w5.jsonl` 같은 보조 산출물은 기본 pool 이 아니라서 뺀다.
            globbed += sorted(p for p in d.glob(f"{doc_id}__*.jsonl")
                              if p.name.count(".") == 1)

    mapping: dict[str, list[dict[str, Any]]] = {}
    ordered_list: list[list[dict[str, Any]]] = []

    for p in exact + globbed:
        if p.exists():
            rows = read_jsonl(p)
            for r in rows:
                c_id = r.get("claim_id")
                results = r.get("results", [])
                if c_id:
                    mapping[c_id] = results
                    suffix = claim_suffix(c_id)
                    # 꼬리 키는 정확한 claim_id 를 덮어쓰지 않게 뒤에서만 채운다.
                    if suffix and suffix not in mapping:
                        mapping[suffix] = results
                ordered_list.append(results)
            if mapping or ordered_list:
                break
    return mapping, ordered_list


def extract_annotator_from_sheet(sheet_name: str) -> str:
    """시트 이름에서 등록된 작업자(시나/윤서/지원/은채/예린) 이름을 추출한다."""
    for name in KNOWN_ANNOTATORS:
        if name in sheet_name:
            return name
    return "시나"


def parse_excel_annotations(excel_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Excel 시트들을 순회하며 각 덱 키워드와 매핑된 라벨 리스트를 생성한다."""
    try:
        import pandas as pd
    except ImportError:
        return {}

    excel_file = pd.ExcelFile(excel_path)
    parsed_data: dict[str, list[dict[str, Any]]] = {}

    for sheet in excel_file.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet)
        if df.empty:
            continue
        
        annotator_name = extract_annotator_from_sheet(sheet)

        # 1. 시트명에서 구체적인 덱 식별자 추출
        # 예전에 여기 `tech_03 중복 시트는 tech_04 다` 라는 예외가 있었다.
        # 그때 워크북에서 지원님이 STRM 덱(tech_03)을 두 시트에 중복
        # 라벨링했고, 그 사본에 이름만 tech_04 로 갈아 끼워 test 로 보냈다.
        # 결과적으로 train 의 tech_03 69행이 test 에 그대로 다시 들어가
        # 모델이 외운 것을 평가하게 됐다. 지금 워크북은 시트가
        # `tech_04__지원`(연소 불안정 75행)으로 제대로 채워져 있으므로
        # 예외 없이 시트 이름을 그대로 믿는다.
        if "arts_01" in sheet:
            # 파일럿 덱(arts_01)은 5명 중 시나 님의 라벨을 대표로 사용
            if annotator_name == "시나":
                deck_key = "arts_01"
            else:
                continue
        else:
            match = re.search(r"(arts_?\d+|bio_?\d+|socio_?\d+|tech_?\d+|finance_?\d+|fin_?\d+)", sheet, re.IGNORECASE)
            if match:
                raw_key = match.group(1).lower().replace("_", "")
                m_num = re.search(r"(\d+)", raw_key)
                if m_num:
                    num = int(m_num.group(1))
                    prefix = re.sub(r"\d+", "", raw_key)
                    if prefix == "fin":
                        prefix = "finance"
                    deck_key = f"{prefix}_{num:02d}"
                else:
                    deck_key = raw_key
            else:
                deck_key = sheet

        records = df.to_dict(orient="records")
        for r in records:
            r["_annotator"] = annotator_name
            r["_sheet_source"] = sheet
        
        parsed_data[deck_key] = records

    return parsed_data


def find_matching_labels(root: Path, deck_id: str, excel_map: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], str, str]:
    """해당 deck_id 에 대한 라벨 데이터 및 작업자 이름을 탐색한다."""
    doc_id = deck_id.split("__")[0].lower()
    
    # 1. 엑셀 파싱 맵에서 직접 일치 또는 부분 일치 검색
    for k, rows in excel_map.items():
        clean_k = k.replace("_", "").lower()
        clean_doc = doc_id.replace("_", "").lower()
        if clean_k == clean_doc or clean_k in clean_doc or clean_doc in clean_k:
            annotator = rows[0].get("_annotator", "시나") if rows else "시나"
            return rows, "excel_sheet", annotator

    # 2. labels/ 디렉터리 내 jsonl 파일 탐색
    label_dir = root / "labels"
    if label_dir.exists():
        matches = list(label_dir.glob(f"{deck_id}*.jsonl")) or list(label_dir.glob(f"{doc_id}*.jsonl"))
        if matches:
            target_file = matches[0]
            # 파일명에 적힌 작업자명 확인 (예: arts_01__claudecode__예린.jsonl)
            annotator = "시나"
            for name in KNOWN_ANNOTATORS:
                if name in target_file.stem:
                    annotator = name
                    break
            return read_jsonl(target_file), "label_jsonl", annotator

    # 3. claims/ 디렉터리 (라벨이 아직 없는 경우 fallback)
    default_annotator_by_deck = {
        "arts_01": "시나",
        "socio_01": "윤서",
        "socio_02": "윤서",
        "bio_01": "예린",
        "bio_02": "예린",
        "bio_03": "은채",
        "bio_04": "예린",
        "tech_01": "지원",
    }
    fallback_annotator = default_annotator_by_deck.get(doc_id, "시나")

    claim_path = root / "claims" / f"{deck_id}.jsonl"
    if claim_path.exists():
        return read_jsonl(claim_path), "claims_jsonl", fallback_annotator
    for f in (root / "claims").glob("*.jsonl"):
        if f.stem.startswith(doc_id) or f.stem == deck_id:
            return read_jsonl(f), "claims_jsonl", fallback_annotator

    return [], "none", fallback_annotator


def build_dataset(root: Path, manifest_path: Path, out_dir: Path, target_deck: str | None = None, excel_file: Path | None = None):
    manifest = load_manifest(manifest_path)
    
    excel_map = {}
    if excel_file and excel_file.exists():
        excel_map = parse_excel_annotations(excel_file)
        print(f"Excel 파싱된 덱 키 목록: {list(excel_map.keys())}")

    available_claim_files = list((root / "claims").glob("*.jsonl"))
    available_decks = [f.stem for f in available_claim_files]
    
    if target_deck:
        decks_to_process = [target_deck]
    elif manifest:
        decks_to_process = list(manifest.keys())
    else:
        decks_to_process = available_decks

    print(f"총 {len(decks_to_process)}개 덱에 대해 Dataset 조립 시작\n")

    split_buckets: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "test": [],
        "generalization": []
    }

    total_records = 0

    for deck_id in decks_to_process:
        raw_items, source_type, annotator_name = find_matching_labels(root, deck_id, excel_map)
        if not raw_items:
            continue

        m_info = manifest.get(deck_id, {})
        doc_id = m_info.get("doc_id", deck_id.split("__")[0])
        split = m_info.get("split", "train")
        domain = m_info.get("domain", "")
        license_type = m_info.get("license", "")
        source_url = m_info.get("source_url", "")
        lang = m_info.get("lang", "ko")
        tool = m_info.get("tool", "gamma")

        # 도메인 한글 정규화
        if not domain or "?" in domain:
            if doc_id.startswith("socio"):
                domain = "사회학"
            elif doc_id.startswith("arts"):
                domain = "예체능"
            elif doc_id.startswith("bio"):
                domain = "바이오/의료"
            elif doc_id.startswith("tech"):
                domain = "기술/공학"
            elif doc_id.startswith("finance"):
                domain = "재무/경제"

        passages_map = load_passages(root, doc_id)
        retrieval_map, retrieval_ordered = load_retrieval_results(root, deck_id)

        deck_records = []
        slide_counts: dict[int, int] = {}

        for idx, item in enumerate(raw_items):
            raw_slide = item.get("slide_idx") or item.get("Slide #") or item.get("Slide") or 1
            try:
                slide_idx = int(re.sub(r"[^\d]", "", str(raw_slide))) if re.search(r"\d", str(raw_slide)) else 1
            except Exception:
                slide_idx = 1
                
            slide_counts[slide_idx] = slide_counts.get(slide_idx, 0) + 1
            c_num = slide_counts[slide_idx]

            # build_queries.py 와 같은 규약을 쓴다 — 두 파일이 다르면 dataset 과
            # retrieval 을 claim_id 로 조인할 때 에러 없이 0건이 나온다.
            # 0 을 채우는 이유는 문자열 정렬에서 s1, s10, s11, s2 로 섞이지 않게 하려고,
            # 구분자가 _ 하나인 이유는 deck_id 자체가 __ 를 쓰기 때문이다.
            claim_id = item.get("claim_id") or f"{deck_id}_s{slide_idx:02d}_c{c_num:02d}"
            claim_text = item.get("claim_text") or item.get("Claim (PPT)") or item.get("claim") or ""
            slide_title = item.get("slide_title") or item.get("Slide_Title") or item.get("title") or ""
            
            raw_ctx = item.get("slide_context") or item.get("Context (PPT)") or item.get("context") or []
            if isinstance(raw_ctx, str):
                slide_context = [c.strip() for c in raw_ctx.split("|") if c.strip()]
            elif isinstance(raw_ctx, list):
                slide_context = list(raw_ctx)
            else:
                slide_context = []

            # top-5 검색 후보 구간 매칭.
            # claim_id 로 먼저 찾고, 표기가 갈린 경우를 위해 꼬리(sNN_cNN)로 한 번 더 찾는다.
            ret_results = retrieval_map.get(claim_id)
            if ret_results is None:
                suffix = claim_suffix(claim_id)
                if suffix:
                    ret_results = retrieval_map.get(suffix)
            # 순서 매칭은 라벨과 검색 결과의 claim 수가 정확히 같을 때만 쓴다.
            # 개수가 다르면 한 칸씩 밀려 엉뚱한 구간이 근거로 붙는다 — 조용해서 더 위험하다.
            if ret_results is None and len(retrieval_ordered) == len(raw_items) and idx < len(retrieval_ordered):
                ret_results = retrieval_ordered[idx]
            if ret_results is None:
                ret_results = []

            candidates = []
            for r in ret_results[:5]:
                p_id = r.get("passage_id") or r.get("span_id", "")
                p_text = passages_map.get(p_id, {}).get("text", "")
                candidates.append({
                    "span_id": p_id,
                    "text": p_text,
                    "score": r.get("score", 0.0),
                    "rank": r.get("rank", 0)
                })

            label = item.get("label") or item.get("Label") or "근거 있음"
            evidence_span = str(item.get("evidence_span") or item.get("Evidence Text (한 문장)") or item.get("evidence_text") or "")

            found_outside = False
            if label == "근거 있음" and evidence_span and evidence_span != "NaN":
                candidate_span_ids = {c["span_id"] for c in candidates}
                if evidence_span not in candidate_span_ids:
                    found_outside = True

            # 행별 또는 덱별 작업자 이름 확정
            row_annotator = item.get("_annotator") or item.get("annotator") or annotator_name
            if row_annotator.lower() in ["sina", "shina"]:
                row_annotator = "시나"

            record = {
                "doc_id": doc_id,
                "domain": domain,
                "license": license_type,
                "source_url": source_url,
                "lang": lang,
                "split": split,
                "deck_id": deck_id,
                "tool": tool,
                "slide_idx": slide_idx,
                "slide_title": slide_title,
                "slide_context": slide_context,
                "claim_id": claim_id,
                "claim_text": claim_text,
                "candidates": candidates,
                "label": label,
                "evidence_span": evidence_span,
                "found_outside_candidates": found_outside,
                "annotator": row_annotator
            }
            deck_records.append(record)

        if split not in split_buckets:
            split_buckets[split] = []
        split_buckets[split].extend(deck_records)
        total_records += len(deck_records)
        print(f"OK   {deck_id}: {len(deck_records)} claims 결합 완료 (작업자: {annotator_name}, 소스: {source_type}) -> split: {split}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, rows in split_buckets.items():
        if rows:
            out_file = out_dir / f"{split_name}.jsonl"
            write_jsonl(out_file, rows)
            print(f"[저장 완료] {out_file} : {len(rows)}개 레코드")

    print(f"\n전체 {total_records}개 Claim 데이터셋 빌드가 성공적으로 완료되었습니다!")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, default=Path("docs/manifest.csv"))
    parser.add_argument("--deck-id", type=str, help="특정 덱 하나만 빌드할 때 지정")
    parser.add_argument("--excel", type=Path, help="annotation.xlsx 파일 지정 (옵션)")
    parser.add_argument("--out-dir", type=Path, default=Path("dataset"))
    args = parser.parse_args()

    build_dataset(args.root, args.manifest, args.out_dir, args.deck_id, args.excel)


if __name__ == "__main__":
    main()