# 실험 프로토콜

## 공정 비교 원칙

Raw와 q는 seed, few-shot split, minibatch 순서, optimizer, scheduler, 학습 step을 동일하게 둡니다. 유일한 차이는 q run에서 optimizer update 이후 element-wise gate를 적용한다는 점입니다. Validation 표본은 학습이나 q 추정에 사용하지 않습니다.

## 기본 설정

| 항목 | 값 |
|---|---:|
| Dataset | FGVC Aircraft base-to-new |
| Shots | class당 16 |
| Batch size | 32 |
| Trainable parameters | CLIP vision/text LayerNorm |
| Initial learning rate | 2e-4 |
| Minimum learning rate | 1e-6 |
| q initialization | 200 images |
| q online samples | 4 images/step |
| q EMA beta | 0.95 |
| q gate maximum | 0.5 |
| Validation EMA samples | Base 16 + New 16/step |
| Validation EMA beta | 0.97 |
| Full validation | 200 steps |
| Checkpoint | 100 steps |

## Validation stream

각 validation 이미지는 한 순환에서 중복 없이 한 번씩 사용한 뒤 새 순환을 시작합니다. 순서는 매 순환 다시 섞되, 직전 순환의 정답/오답 비율이 prefix마다 지나치게 치우치지 않도록 배치합니다. 시작 EMA는 전체 validation 1회 결과로 초기화합니다.

## 저장 항목

- 매 step: loss, Base/New query accuracy, EMA Base/New/HM, q/gate 요약 통계, 시간
- 200 step: 전체 validation Base/New/HM
- q run: parameter metadata와 step별 q vector chunk
- checkpoint: trainable parameters, optimizer, scheduler, scaler, q moments, EMA, validation stream, RNG state

## 보고 기준

최종 step만 선택하지 않고, 미리 정한 budget에서의 성능과 full-validation curve를 함께 보고합니다. Hyperparameter를 같은 seed 결과로 고른 경우 그 seed는 최종 평가에서 분리해야 합니다. 최소 5개 paired seed의 Base, New, harmonic mean 평균±표준편차를 권장합니다.
