# M1 - frozen embedding + lightweight classifier

## 구현 모델

`frozen 임베딩 + 경량 분류기` 구현 모델

임베딩 모델로는 검색 단계에서 이미 사용하고 있던 `intfloat/multilingual-e5-large`를 선택

### Train 11덱

`arts_01`, `arts_02`, `bio_01`, `bio_02`, `bio_03`, `socio_01`, `socio_02`, `socio_03`, `tech_01`, `tech_02`, `tech_03`

### Test 4덱

`arts_03`, `bio_04`, `socio_04`, `tech_04`


## 분류기 선택 과정

데이터 수와 임베딩 차원을 고려해 아래 세 종류를 다시 비교했다.

- Logistic Regression
- Linear SVM
- RBF SVM

클래스 분포는 `근거 있음 529 / 무근거 61 / 모순 78 / Benign 127`로 불균형하다. 그래서 `class_weight=None`과 `class_weight=balanced`를 모두 후보에 넣었다. `C`도 분류기별로 여러 값을 비교했다. 총 22개 조합을 실험했다.

하이퍼파라미터를 고를 때 test는 사용하지 않았다. Train 데이터를 덱 단위로 나눈 5-fold GroupKFold를 사용했고, 평균 macro-F1이 가장 높은 조합을 선택했다. Claim을 무작위로 나누는 대신 덱 단위로 나눈 이유는 같은 덱의 표현 방식이 학습과 validation에 동시에 들어가 성능이 부풀려지는 일을 줄이기 위해서다.

각 분류기에서 가장 좋았던 조합은 다음과 같다.

| 분류기 | 설정 | Train CV macro-F1 |
| --- | --- | ---: |
| Linear SVM | `C=1.0`, `class_weight=balanced` | **0.3309** |
| Logistic Regression | `C=10.0`, `class_weight=balanced` | 0.3292 |
| RBF SVM | `C=10.0`, `class_weight=balanced` | 0.3137 |

Linear SVM이 가장 높아서 최종 M1 분류기로 선택했다. 다만 Linear SVM과 Logistic Regression의 차이는 0.0017로 매우 작다. 이번 결과만으로 Linear SVM이 항상 더 좋다고 말하기보다는, 현재 train split에서 근소하게 앞선 모델이라고 해석하는 편이 맞다.

```bash
python src/train_m1.py --local-files-only
```

* 기본값은 CPU, batch size 16, seed 42다. Batch size는 frozen encoder의 임베딩 추출 묶음 크기라 모델 학습 방향을 정하는 하이퍼파라미터는 아니다. 메모리가 충분하면 `--batch-size 32`처럼 높여 실행 시간을 줄일 수 있다.
