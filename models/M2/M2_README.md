https://colab.research.google.com/drive/1WSI7gVbxvmMGtdTzY6Kaf-e47sKXMgcD?usp=sharing
코랩 환경으로 실행했습니다. 

기존 계획서에 명시된 mDeBERTa-v3-base-mnli-xnli 모델 대신, 한국어 처리에 최적화된 klue/roberta-base 모델을 채택했습니다.
초기 지정된 mDeBERTa 모델을 적용해 본 결과, 파이토치 어텐션 연산 과정에서 CUDA 에러가 계속 발생하고 loss가 nan으로 나오는 오류가 있어 KLUE-RoBERTa로 변경했습니다.