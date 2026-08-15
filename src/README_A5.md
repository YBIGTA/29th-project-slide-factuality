# A-5 · 임베딩 검색기 (dense retriever)

`arts_01` 로 만들고 검증했다. claim 116개 × passage 69개.

## 실행

```bash
python -m pip install -r requirements.txt

# ① spans -> passage pool  (BM25 도 이걸 읽는다)
python src/build_passages.py --all

# ② claims -> claim_id 가 붙은 쿼리 파일  (BM25 도 이걸 읽는다)
python src/build_queries.py --all

# ③ 임베딩 검색
python src/dense_retrieve.py --all --top-n 20

# ④ 자체 점검
python src/eval_retrieval.py --deck-id arts_01__claudecode
```

산출물

| 경로 | 내용 | 누가 쓰나 |
| --- | --- | --- |
| `passages/{doc_id}.jsonl` | passage pool (w3 기준) | **A-4(BM25) · A-5 공용** |
| `passages/{doc_id}.w5.jsonl` | w5 pool (선택) | 판별 모델이 넓은 문맥을 원할 때 |
| `retrieval/queries/{deck_id}.jsonl` | claim_id 가 붙은 claim | **A-4 · A-5 공용** |
| `retrieval/dense/{deck_id}.jsonl` | claim 별 top-20 | RRF 융합 |
| `.cache/emb/*.npy` | passage 임베딩 캐시 (gitignore) | |

---

## 팀 계약 — 여기만 지키면 융합이 된다

BM25 담당자는 `claims/*.jsonl` 을 **직접 읽지 말고** `retrieval/queries/{deck_id}.jsonl`
을 읽어라. claim_id 가 자동으로 일치한다. 결과는 아래 스키마로
`retrieval/bm25/{deck_id}.jsonl` 에 쓰면 융합 단계가 `retriever` 필드만 보고
그대로 RRF 를 돌린다.

```json
{"claim_id": "arts_01__claudecode_s03_c05",
 "deck_id": "arts_01__claudecode",
 "doc_id": "arts_01",
 "retriever": "dense_e5",
 "model": "intfloat/multilingual-e5-large",
 "window": "w3",
 "query_text": "안달루시아 음악 속 수천 년의 공생",
 "results": [{"rank": 1, "passage_id": "arts_01_w3_002", "score": 0.8712,
              "sent_ids": ["s002", "s003", "s004"]}]}
```

### ID 규칙

```
passage_id = "{doc_id}_{span_id}"                       arts_01_w3_007
claim_id   = "{deck_id}_s{슬라이드:02d}_c{순번:02d}"      arts_01__claudecode_s03_c05
```

`passage_id` 는 A-2 의 `span_id` 에 `doc_id` 만 붙인 것이다. 되돌릴 수 있어야
하기 때문이다 — B-1 의 정답 집합은 원래 `span_id` 로 되어 있어서,
`passage_id[len(doc_id)+1:]` 로 떼면 그대로 맞출 수 있다.

`{doc_id}_p7` 같은 형태를 안 쓴 이유는 `p7` 이 어떤 청킹으로 만든 7번인지를
잃어버리기 때문이다. `w3_007` 은 청킹 단위가 ID 안에 남아 w3/w5 를 섞어도
충돌하지 않는다.

`claim_id` 의 순번은 **슬라이드 안에서만** 센다. 파일 전체 줄번호로 세면
앞 슬라이드에 claim 이 하나 늘 때 뒤가 전부 밀린다. `claims/*.jsonl` 은
이미 통째로 재생성된 적이 있다(커밋 a0d8095).

### RRF 할 때 주의

**score 를 더하지 마라. rank 만 써라.** cosine(0.70~0.90)과 BM25 점수는
스케일이 완전히 다르다. score 필드는 디버깅용으로만 실어 보낸다.

---

## 설계 결정과 이유

### passage 를 새로 자르지 않고 A-2 의 w3 구간을 그대로 썼다

`README_A2.md` 가 이미 "A-4 는 구간을 색인, A-5 는 구간에 `passage:` 접두사,
B-1 은 `find_evidence(...).span_ids` 가 정답"이라고 인계하고 있다. 여기서
독자적으로 청킹하면 B-1 이 recall 을 계산할 수 없다.

w3 를 기본으로 잡은 건 길이 때문이다.

| pool | 구간 수 | 길이 중앙값 | 최대 |
| --- | --- | --- | --- |
| w3 | 69 | 249자 | 409자 |
| w5 | 67 | 433자 | 593자 |

판별 모델(mDeBERTa-v3-base)이 512 토큰이다. w5 최대 593자에 claim 까지 붙이면
잘릴 수 있다. w5 는 `--window w5` 로 따로 뽑아 두었으니 판별 모델 쪽에서
필요하면 쓰면 된다.

### 구간이 겹치는 건 버그가 아니다

stride 가 1이라 한 문장이 w3 에서 3개 구간에 들어간다. 그래서 top-N 에
내용이 거의 같은 구간이 연달아 뜬다. A-2 의 설계이고 정답도 **집합**이다.

→ **RRF 이후 판별 모델에 넘기기 전에 `sent_ids` 겹침 기준으로 합치는 단계가
필요하다.** 그래서 결과마다 `sent_ids` 를 같이 실어 보낸다. 안 합치면
top-5 가 사실상 같은 문장 3~4개를 중복해서 보게 된다.

### 모델은 `intfloat/multilingual-e5-large`

1024차원, 512토큰. `passage: ` / `query: ` 접두사를 붙여 학습된 모델이라
접두사를 빼면 성능이 떨어진다. `-instruct` 계열은 접두사 규약이 달라서
(instruct 템플릿) 팀 표준과 어긋나므로 쓰지 않았다.

코퍼스가 문서당 70구간 수준이라 M1 Mac(mps)에서 전체 30초면 끝난다.
passage 임베딩은 내용 해시로 캐시하므로 두 번째부터는 claim 인코딩만 한다.

검색은 **같은 doc_id 안에서만** 한다. 문서 간 교차 검색은 하지 않는다.

---

## 이 데이터에서 실제로 확인한 것

### 1. claim 의 3분의 2는 원문과 어휘가 안 겹친다

`find_evidence` 로 116개 claim 을 원문에 직접 대조한 결과다.

| | 개수 |
| --- | --- |
| exact (원문에 그대로 있음) | 33 |
| **none (어휘 일치 없음)** | **83** |

그리고 exact 33개는 거의 전부 `1961`, `카이로`, `파리`, `릴리 라바시` 같은
**짧은 고유명사·연도**다. 긴 claim 은 전부 AI 가 다시 쓴 문장이라 어휘가
겹치지 않는다.

→ **BM25 단독으로는 83개를 못 잡는다.** 반대로 dense 는 `1961` 같은
2~4자 토큰에 약하다. RRF 융합이 필요한 이유가 이 표에 그대로 나온다.

### 2. claim 이 짧을수록 검색이 흐려진다

전체 116개 claim 의 top-1 cosine 중앙값:

| claim 길이 | 개수 | top-1 cosine |
| --- | --- | --- |
| 10자 미만 | 42 (36%) | 0.774 |
| 10–29자 | 40 | 0.837 |
| 30자 이상 | 34 | 0.871 |

10자 미만이 36%나 되는데(`핵심 메커니즘`, `+0.048***`, `현재 불평등` 등),
`guideline_v2.md` 는 고유명사·연도도 Benign 으로 빼지 말고 반드시
근거있음/무근거/모순으로 판별하라고 규정한다. 즉 이 42개도 검색 대상이다.
**여기가 BM25 가 메워야 할 구간이다.**

### 3. cosine 절대값은 아무 의미가 없다

`발표 구성`(목차 라벨, 원문에 있을 리 없는 텍스트)도 top-1 cosine 이 0.74 다.
e5 는 점수가 좁은 구간에 몰린다. **"cosine 0.8 이상이면 근거 있음" 같은
임계값을 세우면 안 된다.** 순위만 신뢰하고, 근거 유무 판정은 판별 모델에 맡긴다.

---

## 자체 점검 결과 (`eval_retrieval.py`)

라벨 시트가 아직 레포에 없어서 **진짜 recall 은 못 잰다.** 대신 claim 이
원문에 그대로 있는 27개만 골라 하한선을 쟀다.

```
recall@1   0.741
recall@5   0.963
recall@10  1.000
recall@20  1.000
MRR        0.835
```

이 27개는 어휘가 일치하는 claim 들이라 **BM25 에 유리하고 dense 에 불리한
쪽으로 편향된 부분집합**이다. 즉 위 숫자는 하한선이지 성능 보고용이 아니다.
top-20 안에 100% 들어온다는 것만 확인된 셈이고, RRF 여유분으로 top-20 을
잡은 근거가 된다.

**라벨 시트(`evidence_text`)가 들어오면 반드시 다시 재야 한다.**

---

## 다음 사람에게 넘길 때

- **A-4 (BM25)** — `retrieval/queries/`, `passages/` 를 읽고 같은 스키마로
  `retrieval/bm25/` 에 쓴다. 위 "팀 계약" 절만 보면 된다
- **RRF 융합** — rank 만 쓴다. 융합 후 `sent_ids` 겹침으로 dedupe 해야 한다
- **B-1 (recall)** — `passage_id[len(doc_id)+1:]` 로 span_id 를 떼서
  `find_evidence(...).span_ids` 와 맞춘다
- **판별 모델** — `passage_id` 로 `passages/{doc_id}.jsonl` 에서 원문을 되찾는다

## 아직 안 된 것

`arts_01` 하나만 돌아간다. 나머지는 passage pool 이 없어서 `dense_retrieve.py`
가 건너뛴다 (쿼리 파일은 5개 덱 모두 만들어져 있다).

| doc | 막힌 이유 |
| --- | --- |
| socio_02 | `docs/clean` 미생성 (단단이라 A-2 파이프라인 그대로 가능) |
| socio_01 | **2단 조판** — `pdf_to_text.py` 가 좌우 단을 한 줄에 섞는다 |
| tech_02, tech_03 | **2단 조판** + 짝이 되는 덱 없음 |
| tech_01, bio_01 | **원문 PDF 자체가 없음** (claim 은 각각 90, 95개 있다) |

단 분리 처리를 A-2 에 붙이면 socio_01 · tech_02 · tech_03 이 한 번에 풀린다.
