# Gradient Sharing Gate

Fine-tuning 중 **한두 이미지에만 국소적으로 반응하는 gradient 요소는 줄이고, 여러 이미지에 걸쳐 영향을 공유하는 요소는 유지**하는 연구용 코드입니다. Gradient를 새로 생성하지 않고, 원래 optimizer가 만든 update를 per-parameter sharing score `q`로 조절합니다.

## 핵심 아이디어

각 이미지의 gradient를 `g_i`라 할 때 parameter별 통계는 다음과 같습니다.

```text
q = E[|g_i|]^2 / (E[g_i^2] + eps)
gate = clamp(q / q_max, 0, 1)
```

- 여러 이미지가 비슷한 크기로 영향을 주면 `q`가 1에 가까워집니다.
- B개 이미지 중 하나만 영향을 주면 `q`는 대략 `1/B`입니다.
- 부호 충돌은 처벌하지 않습니다. 절댓값을 사용하므로 서로 반대 방향인 유효한 신호도 공유성으로 인정합니다.
- gate는 gradient 자체를 예측하지 않고, optimizer가 제안한 parameter displacement만 요소별로 축소합니다.

## 현재 실험

- Model: pretrained CLIP
- Adaptation: vision/text encoder의 LayerNorm만 학습
- Dataset: FGVC Aircraft, base-to-new 16-shot
- Batch size: 32
- q 초기화: 200개 이미지
- q 온라인 갱신: step당 4개 이미지, EMA beta 0.95
- gate: `q < 0.5`를 0~1로 선형 변환
- validation EMA: step당 Base 16 + New 16, beta 0.97
- full validation: 200 step마다

Seed 1의 예비 결과는 [results/fgvc_seed1](results/fgvc_seed1)에 포함했습니다. 현재 한 seed에서는 q가 raw보다 뚜렷하게 우수하다고 말할 수 없습니다. 이 레포는 아이디어를 검증하고 실패 원인을 분리하기 위한 재현 가능한 기반입니다.

## 설치

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

FGVC Aircraft는 기본적으로 `data/fgvc_aircraft/`에서 찾고, 없으면 torchvision이 자동으로 내려받습니다.

## 실행

같은 seed의 raw와 q를 반드시 짝지어 비교합니다.

```powershell
./scripts/run_fgvc_raw.ps1 -Seed 1
./scripts/run_fgvc_q.ps1 -Seed 1
python scripts/plot_results.py
```

긴 실험은 `--checkpoint`, `--checkpoint_every`, `--resume`을 이용해 이어갈 수 있습니다. 구체적인 설정은 [실험 프로토콜](docs/experiment_protocol.md), 연구 의도와 주의점은 [연구 방향](docs/research_direction.md)을 참고하세요.

## 구조

```text
gradient_sharing_gate/   q 통계, CLIP wrapper, FGVC 데이터 구성
experiments/             CLIP 2SFS Stage-1 학습 실행 파일
configs/                 확정한 실험 설정
scripts/                 raw/q 실행, 결과 요약 및 그래프
results/                 작은 CSV와 요약만 버전 관리
docs/                    가설, 실험 설계, 한계
tests/                   핵심 수식 단위 테스트
```

대용량 데이터, 체크포인트, step별 q vector는 Git에 넣지 않도록 설정되어 있습니다.
