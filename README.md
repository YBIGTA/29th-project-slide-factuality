# 문서–슬라이드 변환의 사실성 검증

AI 가 논문으로 만든 발표 슬라이드가 **원문에 근거하는가**를 판정한다.

세상의 진실이 아니라 **주어진 원문**이 기준이다. 슬라이드에 적힌 말이
역사적 사실이더라도 원문 지면에 없으면 `무근거` 다. 그래서 어노테이터도
판정 모델도 **인터넷 검색을 쓰지 않는다.**

라벨은 4분류다.

| 라벨 | 뜻 |
| --- | --- |
| `근거 있음` | 원문에 있거나, 원문을 논리적 비약 없이 요약한 것 |
| `무근거` | 원문에서 확인되지 않는 것 (과도한 구체화 포함) |
| `모순` | 원문의 사실·수치·저자 주장과 정면으로 충돌하는 것 |
| `Benign` | 팩트체크 대상이 아닌 것 ("감사합니다", "Q&A") |

기준의 세부는 [`guides/guideline_v2.md`](guides/guideline_v2.md) 에 있다.
판정이 갈리면 그 문서가 우선한다.

---

## 데이터가 흐르는 순서

```
docs/raw/*.pdf                 원문 PDF
      │  pdf_to_text.py         (초벌 변환 — 사람이 검수한다)
      ▼
docs/clean/{doc_id}.txt        정제 텍스트
      │  segment.py             A-2
      ▼
spans/sents · w3 · w5          문장 / 3문장 구간 / 5문장 구간
      │  build_passages.py      A-5
      ▼
passages/{doc_id}.jsonl        검색기 공용 입력 (문서 간 고유 ID)


decks/*.pptx                   AI 가 만든 발표 덱
      │  claim_split.py         A-3
      ▼
claims/{deck_id}.jsonl         검증 대상 문장(claim)
      │  build_queries.py       A-5
      ▼
retrieval/queries/*.jsonl      claim_id 가 붙은 검색 쿼리
      │
      ├─ retrieve_bm25.py  ──►  retrieval/bm25/*.jsonl      A-4
      └─ dense_retrieve.py ──►  retrieval/dense/*.jsonl     A-5
                                        │
                                        │  + 사람이 붙인 라벨
                                        ▼  build_dataset.py  B-2
                                 dataset/{split}.jsonl
```

`doc_id` 는 원문(`arts_01`), `deck_id` 는 그 원문으로 만든 덱
(`arts_01__claudecode`)이다. 둘의 연결은 `docs/manifest.csv` 에만 적혀 있다.

---

## 폴더

| 경로 | 무엇이 들어 있나 |
| --- | --- |
| `docs/manifest.csv` | **모든 작업의 기준 파일.** 원문↔덱↔split 대응. 새 자료는 여기 먼저 등록한다 |
| `docs/raw/` | 원문 PDF 원본. 손대지 않는다 |
| `docs/clean/` | 정제 텍스트. `.review.txt` 는 무엇을 지웠는지의 기록이라 **눈으로 확인해야 한다** |
| `decks/` | AI 가 생성한 PPTX. 파일명 = `deck_id` |
| `claims/` | 덱에서 뽑은 claim. 판정 단위 |
| `spans/` | 문장(`sents/`)과 근거 구간(`w3/`, `w5/`). 구간은 stride 1 로 겹친다 |
| `passages/` | `spans/` 에 문서 간 고유 ID 를 붙인 것. 검색기가 읽는다 |
| `retrieval/queries/` | claim 에 `claim_id` 를 부여한 파일. BM25 와 임베딩이 **같은 것을** 읽는다 |
| `retrieval/bm25/`, `retrieval/dense/` | 검색 결과. 스키마가 같아서 RRF 로 융합할 수 있다 |
| `annotation/` | 사람이 붙인 라벨 |
| `dataset/` | 학습·평가용 최종 데이터 |
| `results/` | 측정 결과 (IAA, 검색 비교) |
| `prompts/` | 덱 생성·claim 분할에 쓴 프롬프트 |
| `guides/` | 어노테이션 가이드라인 |
| `src/` | 코드 |

---

## `src/` 파일별 역할

| 파일 | 담당 | 하는 일 |
| --- | --- | --- |
| `pdf_to_text.py` | A-2 | PDF → 초벌 텍스트. 머리말·각주를 걷어내고 줄바꿈으로 쪼개진 어절을 붙인다. **출력은 반드시 사람이 검수한다** |
| `segment.py` | A-2 | 문장 분리 + 구간 생성. `find_evidence()` 로 라벨의 근거 문장을 구간에 붙인다 |
| `test_segment.py` | A-2 | 마침표 예외·근거 정렬 테스트. 새 논문에서 이상하게 잘리면 케이스를 여기 추가한다 |
| `claim_split.py` | A-3 | PPTX → claim JSONL. 표·차트·그룹까지 훑고 항목마다 태그(제목/불릿/행…)를 붙인다 |
| `check_manifest.py` | — | `manifest.csv` 점검. 기계로 알 수 있는 값은 채우고 빠진 항목을 알려준다 |
| `build_passages.py` | A-5 | 구간에 문서 간 고유 `passage_id` 를 붙인다 |
| `build_queries.py` | A-5 | claim 에 `claim_id` 를 **한 번만** 부여한다. 두 검색기가 결과를 맞출 수 있는 근거 |
| `retrieve_bm25.py` | A-4 | 형태소 토큰 기반 BM25. 공백 분리하면 `정책을/정책이` 가 갈라져서 형태소를 쓴다 |
| `dense_retrieve.py` | A-5 | multilingual-e5 임베딩 검색. passage 에 `passage:`, claim 에 `query:` 접두사가 필수다 |
| `eval_retrieval.py` | A-5 | 검색기 자체 점검. **진짜 평가가 아니다** — claim 이 원문에 그대로 있는 경우만 정답으로 쓰는 하한선 측정 |
| `pick_extraction.py` | A-2 | PDF 를 1단·2단 두 방식으로 뽑아 점수를 재고 나은 쪽을 남긴다 |
| `compare_retrieval.py` | B-1 | 라벨의 근거로 정답을 만들고 BM25·임베딩·하이브리드 3종을 비교한다 |
| `stamp_annotation.py` | — | 라벨 워크북의 각 시트에 `deck_id`·`담당자` 를 박고, 요약 시트에 원문·덱이 레포에 있는지를 표시한다 |
| `summarize_b1.py` | B-1 | 덱별 지표를 모아 `results/retrieval_summary.md` 한 표로 만든다 |
| `build_dataset.py` | B-2 | 라벨 + 검색 결과 + claim 을 합쳐 `dataset/*.jsonl` 로. `manifest.csv` 의 split 을 따른다 |
| `pipeline_test.py` | — | A-2 → A-3 → 검색 → B-2 를 한 번에 돌려보는 통합 점검 |
| `README_A2.md` | | 문장 분리기 상세 — 실제 논문에서 터진 예외 6가지 |
| `README_A5.md` | | 임베딩 검색기 상세와 BM25 와의 "팀 계약" |
| `README_B1.md` | | 검색 방식 비교 — 정답 만드는 법, 가중 RRF, 지표 정의 |

---

## 실행

```bash
pip install -r requirements.txt

# 원문 한 건을 끝까지
python src/pdf_to_text.py docs/raw/arts_01.pdf --doc-id arts_01
#   -> docs/clean/arts_01.review.txt 를 보고 손으로 고친 뒤 다음으로
python src/segment.py docs/clean/arts_01.txt --doc-id arts_01
python src/build_passages.py --doc-id arts_01

# 덱 한 건
python src/claim_split.py decks/arts_01__claudecode.pptx --output-dir claims
python src/build_queries.py --deck-id arts_01__claudecode
python src/retrieve_bm25.py  --deck-id arts_01__claudecode --top-n 20
python src/dense_retrieve.py --deck-id arts_01__claudecode --top-n 20

# 테스트는 레포 루트에서 (docs/clean 을 상대경로로 읽는다)
python src/test_segment.py
```

---

## 알아둘 것

**구간은 겹친다.** stride 가 1 이라 한 문장이 w3 에서는 3개, w5 에서는 5개
구간에 들어간다. 정답은 단일 구간이 아니라 **집합**이고, top-k 안에 그 중
하나라도 있으면 hit 다. 정답을 하나로 잡으면 recall 이 실제보다 낮게 나온다.

**근거는 원문에서 복붙한다.** 라벨 시트의 근거 문장은 손으로 타이핑하거나
요약하면 안 된다. `find_evidence()` 가 원문과 대조해서 구간에 붙이는데,
`...` 로 중간을 생략하거나 `(p.131)` 같은 메모를 덧붙이면 정렬에 실패하고
그 라벨은 검색기 평가에서 통째로 빠진다.

**claim 분할기 출력이 바뀌면 라벨이 어긋난다.** 같은 덱이라도 분할기 버전이
다르면 claim 목록이 달라진다. 실제로 `arts_01` 4번 슬라이드와 `socio_01` ·
`socio_02` 에서 어긋난 이력이 있다. 라벨링을 시작하기 전에 `claims/` 를
고정하고, 바꿔야 하면 전원에게 알린다.

---

## 지금 손봐야 하는 것

| | 문제 |
| --- | --- |
| `claim_id` 규약 충돌 | `build_queries.py` 는 `arts_01__claudecode_s01_c01`, `build_dataset.py` 는 `socio_01__claudecode__s1__c1` 을 만든다. `dataset/` 과 `retrieval/` 을 `claim_id` 로 조인하면 **에러 없이 0건**이 나온다 |
| `manifest.csv` BOM | 파일 앞에 UTF-8 BOM 이 있는데 `build_queries.py` 는 `encoding="utf-8"` 로 연다. 첫 컬럼명이 `'﻿doc_id'` 가 되어 **등록된 8개 덱이 전부 매핑 0건**이다. 읽는 쪽을 `utf-8-sig` 로 고쳐야 한다 |
| `manifest.csv` 한글 깨짐 | `socio_01` · `socio_02` · `arts_01` 행의 제목이 CP949 로 저장돼 깨져 있다 (`蹂듭??쒕룄`) |
| `dataset/train.jsonl` 스키마 | `candidates` · `evidence_span` · `found_outside_candidates` 545행이 전부 비어 있다. 수동 라벨링으로 바꾸면서 죽은 필드이고, 실제로 채워지는 건 `evidence_text` 다 |
| `claims/bio_02__claudecode__X1.jsonl` | 변형본인데 원본 `bio_02__claudecode.jsonl` 과 바이트가 같다. X1 덱으로 다시 뽑아야 한다 |
| 원문 없는 라벨 | `arts_02` · `arts_03` · `socio_03` · `socio_04` · `fin_01` · `fin_02` 는 라벨만 있고 원문·덱·manifest 항목이 없다. `tech_02` · `tech_03` 은 PDF 만 있다 |
