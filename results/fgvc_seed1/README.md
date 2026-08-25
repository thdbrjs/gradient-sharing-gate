# FGVC Aircraft — Seed 1 legacy preliminary result

Full validation은 200 step마다 측정했습니다. 이 결과는 이전 설정인 `q_abs`와 `gate=clamp(q/0.5, 0, 1)`로 얻었으며, 현재 기본 설정인 `q_signed`와 `gate=q`의 결과가 아닙니다.

| Run | Steps | Best HM (step) | Base | New | Final HM |
|---|---:|---:|---:|---:|---:|
| Raw | 3000 | 39.13% (2200) | 40.97% | 37.45% | 38.97% |
| q gate | 5000 | 38.99% (1400) | 39.59% | 38.42% | 38.92% |

이 결과만으로 q gate의 개선을 주장할 수 없습니다. Raw와 q의 최고점 차이는 작고 q는 더 긴 budget을 사용했습니다. 현재 데이터는 구현 검증과 후속 ablation의 기준선으로 취급합니다.
