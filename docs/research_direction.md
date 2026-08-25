# 연구 방향

## 문제 정의

일반 fine-tuning은 관측된 base 데이터의 loss만 줄입니다. 따라서 어떤 parameter update가 여러 이미지와 class에 공통으로 작용하는지, 혹은 특정 이미지에만 맞춰진 변화인지 구분하지 않습니다. 이 연구는 base에서 얻은 변화가 다른 base 이미지와 궁극적으로 new class에도 전파되려면, **개별 이미지에 고립된 update보다 여러 이미지가 공유하는 update를 우선해야 한다**는 가설에서 출발합니다.

## 제안 방식

Gradient generator가 새로운 gradient를 출력하게 하지 않습니다. 원래 loss와 optimizer가 제안한 update는 보존하고, 각 parameter 요소의 cross-example sharing score를 추정하여 update 크기만 0~1 사이에서 조절합니다.

현재 기본 실험은 방향 합의를 보는 다음 점수를 사용합니다.

`q_signed = E[g]^2 / E[g^2]`

비교군인 `q_abs = E[|g|]^2 / E[g^2]`는 부호와 무관한 관여 범위를 측정합니다.

`signed`에서는 이미지 간 부호가 충돌하면 평균이 상쇄되어 q가 작아집니다. 따라서 여러 이미지가 해당 요소를 사용할 뿐 아니라 같은 update 방향에 동의하는지를 요구합니다. Gate는 별도 재조정 없이 계산된 q의 0~1 범위를 그대로 사용합니다.

## 현재 주장의 범위

q는 parameter가 여러 **학습 이미지의 loss**에 관여하는지를 측정합니다. Parameter update가 실제 embedding을 얼마나 움직이는지 또는 new class에 일반화되는지를 직접 측정하는 값은 아닙니다. 따라서 다음 연결은 실험으로 검증해야 합니다.

```text
학습 이미지 간 gradient 공유성
→ representation 변화의 공유성
→ new-class 성능 보존 또는 향상
```

## 예상되는 실패 원인

- q가 중요도가 아니라 단순한 빈도만 측정할 수 있습니다.
- 자주 등장하지만 해로운 gradient도 높은 q를 가질 수 있습니다.
- 4개 이미지의 온라인 표본은 분산이 크고, EMA가 변화에 늦게 반응할 수 있습니다.
- q가 빠르게 포화되면 gate 대부분이 1이 되어 raw와 차이가 사라집니다.
- LayerNorm만 학습할 때와 전체/LoRA 학습일 때 q의 의미가 다를 수 있습니다.
- class-balanced batch가 일반 minibatch 분포와 다른 통계를 만들 수 있습니다.

## 다음 검증 순서

1. Raw와 q를 같은 seed·batch 순서로 paired comparison합니다.
2. 여러 seed의 평균과 표준편차를 보고합니다.
3. random gate, magnitude-only gate, fixed layer gate를 대조군으로 둡니다.
4. q 구간별 실제 parameter displacement와 Base/New 성능 기여를 측정합니다.
5. q와 embedding 변화 공유성의 상관을 별도 진단 실험으로 확인합니다.
6. FGVC 외 작은 class 수의 데이터셋에서도 반복합니다.
