# M3_smoke — 판정 모델

- **구성**: klue/roberta-base 파인튜닝, 맥락 **포함** (claim + 슬라이드 맥락 + 후보 구간 top-5)
- **백본**: `klue/roberta-base`
- **확인하려는 것**: 슬라이드 맥락의 효과 (← 프로젝트 핵심 주장)
- **git commit**: `1f1ee31`

## 왜 이 백본인가

계획서의 `mDeBERTa-v3-base-mnli-xnli` 대신 M2 담당자가 채택한 백본과 동일하게 맞췄다. M2 에서 mDeBERTa 는 fp16 학습 중 loss=nan 이 재현되어 사용할 수 없었고, **M2' 와 M3 가 백본까지 동일해야 맥락 ablation 이 성립**하기 때문이다.

## 입력 포맷

```
[CLS] claim [SEP] 슬라이드 제목 + 같은 슬라이드 다른 불릿 [SEP] 후보 구간 1..5 [SEP]
```

### 토큰 예산 (상한 510)

| 블록 | 상한 | 비고 |
|---|---|---|
| claim | 64 | 원자 명제 1개 |
| 슬라이드 맥락 | 96 | 제목 + 불릿 6개까지 |
| 후보 × 5 | 64 × 5 = 320 | 3문장 윈도우를 앞 64토큰으로 절단 |
| 특수 토큰 | 8 | |

블록마다 따로 자른다. HF 기본 `truncation='longest_first'` 에 맡기면 어느 블록이 희생되는지 통제할 수 없어 claim 이 밀려 나갈 수 있다.

> `klue/roberta-base` 은 절대 위치 임베딩을 쓰므로 상한을 넘기면 런타임 에러가 난다. 실사용 상한이 512 가 아니라 510~511 이라 `max_len=510` 로 고정했다.

## 하이퍼파라미터

| 항목 | 값 |
|---|---|
| `learning_rate` | 2e-05 |
| `num_train_epochs` | 1.0 |
| `per_device_train_batch_size` | 16 |
| `gradient_accumulation_steps` | 1 |
| `유효 배치` | 16 |
| `weight_decay` | 0.01 |
| `warmup_ratio` | 0.1 |
| `max_grad_norm` | 1.0 |
| `label_smoothing_factor` | 0.0 |
| `lr_scheduler_type` | linear |
| `fp16` | True |
| `클래스 가중치` | 역빈도^0.5 (근거있음 편중 보정) |
| `early stopping` | dev macro-F1 기준, epoch 단위, best 복원 |
| `dev 분할` | train 에서 **문서 단위** 3덱 (시드마다 다름) |
| `시드` | 13 |

## 데이터

| 파일 | sha256(12) | 건수 |
|---|---|---|
| `train.jsonl` | `52380026c991` | 811 |
| `test.jsonl` | `41ebe14d2f03` | 290 |
| `generalization.jsonl` | `4a3f556253bf` | 259 |

라벨 분포 (train): {'근거있음': 528, '모순': 78, '무근거': 61, 'benign': 127}

> 해시를 기록하는 이유: `dataset/` 이 8/19~8/21 계속 갱신되었다. M2' 와 M3 가 같은 스냅샷으로 학습되었는지는 이 해시로만 증명된다.

## 결과 (test)

| 시드 | dev macro-F1 | test macro-F1 | accuracy | CI95 (문서 부트스트랩) |
|---|---|---|---|---|
| 13 | 0.2165 | 0.2066 | 0.7040 | 0.198~0.218 |

**test macro-F1 = 0.2066 ± 0.0000** (1 시드)

주 지표는 accuracy 가 아니라 **macro-F1** 이다. 근거있음이 다수 클래스라 accuracy 로 보고하면 "전부 근거있음으로 찍어도 그 정도"라는 지적이 즉시 들어온다.

### 혼동행렬 (전체 시드 합산, 행=정답 / 열=예측)

| gold \ pred | 근거있음 | 무근거 | 모순 | benign |
|---|---|---|---|---|
| **근거있음** | 195 | 0 | 0 | 0 |
| **무근거** | 26 | 0 | 0 | 0 |
| **모순** | 12 | 0 | 0 | 0 |
| **benign** | 44 | 0 | 0 | 0 |

## 저장된 모델

- 이 폴더의 가중치는 **dev macro-F1 최고 시드(seed 13, 0.2165)** 의 것이다.
- `model.safetensors` 는 용량 때문에 git 에 올리지 않는다 (`.gitignore`). Drive / 로컬에 보관하고 필요하면 아래 명령으로 재현한다.

## 재현

```bash
python src/train_judge_m3.py --model_id M3_smoke --use_context \
       --backbone klue/roberta-base --max_len 510
```

## 관련 산출물

- `results/M3_smoke_test.json` — 시드별 원시 결과
- `results/model_comparison.md` — M1 / M2 / M2' / M3 비교표
- `src/check_ablation.py` — M2' 와 M3 가 정말 `--use_context` 하나만 다른지 기계 검증
