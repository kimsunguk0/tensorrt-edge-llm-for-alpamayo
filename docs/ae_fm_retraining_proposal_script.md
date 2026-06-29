# AE/FM Retraining Proposal Script

## 1. 현재 runtime에서 쓰는 Action Expert / Flow Matching 위치

현재 실차 후보 명령어에서 쓰는 post-VLM trajectory engine은 아래 TensorRT plan이다.

```text
/workspace/models/alpamayo_runtime/fm/flowmatching_20260424_trt_fp8/engine_thor/teacher_structured_student_reflow_consistency_step4_mlp6144_one_step_fp8_s3328_thor.plan
```

이 plan은 `--fm-engine`으로 들어가며, 현재 legacy seed23 no-nav baseline의 Action Expert / Flow Matching 역할을 한다. 즉 현재 baseline에서는 Action Expert와 Flow Matching이 따로 분리된 배포 엔진이 아니라, post-VLM FM one-step TensorRT engine 하나로 묶여 있다.

현재 runtime 조합:

```text
LLM engine:
/workspace/models/alpamayo_runtime/engines/alpa1.5/llm.engine

Visual engine:
/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild/visual/visual.engine

Action Expert / Flow Matching engine:
/workspace/models/alpamayo_runtime/fm/flowmatching_20260424_trt_fp8/engine_thor/teacher_structured_student_reflow_consistency_step4_mlp6144_one_step_fp8_s3328_thor.plan
```

현재 기준 실행 설정:

```text
nav: off
diffusion_seed: 23
diffusion_num_steps: 2
alpamayo_fm_use_prefill_kv: on
```

## 2. 현재 TensorRT plan의 ONNX source

현재 FP8 Thor plan은 아래 ONNX에서 빌드된 것으로 build log에서 확인된다.

```text
ONNX:
/workspace/models/alpamayo_runtime/fm/flowmatching_20260424_deploy/onnx/teacher_structured_student_reflow_consistency_step4_mlp6144_one_step_fp16_s3328.onnx

External weights:
/workspace/models/alpamayo_runtime/fm/flowmatching_20260424_deploy/onnx/teacher_structured_student_reflow_consistency_step4_mlp6144_one_step_fp16_s3328.onnx.data

Constants:
/workspace/models/alpamayo_runtime/fm/flowmatching_20260424_deploy/onnx/teacher_structured_student_reflow_consistency_step4_mlp6144_one_step_fp16_s3328.onnx.constants.json
```

Build command from log:

```bash
/opt/tensorrt/bin/trtexec \
  --onnx=/workspace/models/alpamayo_runtime/fm/flowmatching_20260424_deploy/onnx/teacher_structured_student_reflow_consistency_step4_mlp6144_one_step_fp16_s3328.onnx \
  --saveEngine=/workspace/models/alpamayo_runtime/fm/flowmatching_20260424_trt_fp8/engine_thor/teacher_structured_student_reflow_consistency_step4_mlp6144_one_step_fp8_s3328_thor.plan \
  --fp8 \
  --fp16 \
  --builderOptimizationLevel=5 \
  --memPoolSize=workspace:8192 \
  --verbose
```

주의:

```text
.plan은 학습 불가
.onnx도 보통 fine-tune 시작점으로는 부적합
재학습에는 ONNX export 이전의 PyTorch checkpoint가 필요함
```

외부 학습 서버에서 찾아야 할 checkpoint 키워드:

```text
flow_matching_standard_student_reflow_consistency_4step_full_ft_e12
teacher_structured_student_reflow_consistency_step4_mlp6144
teacher_structured_student_reflow_consistency_step4
reflow_consistency_step4
student_reflow_consistency
```

## 3. 현재 FM engine contract

현재 legacy FM engine은 TensorRT runtime에서 `EngineKind::kLegacyFm`으로 처리된다.

ONNX input:

```text
x               float32 [1, 64, 2]
t               float32 [1, 1, 1]
dt              float32 [1, 1, 1]
kv_cache        float16 [36, 1, 2, 8, 3328, 128]
attention_mask  float32 [1, 1, 64, 3392]
position_ids    int64   [3, 1, 64]
```

ONNX output:

```text
next_x              float32 [1, 64, 2]
v                   float32 [1, 64, 2]
future_token_embeds float32 [1, 64, 2048]
```

Model notes from deploy bundle:

```text
backbone: teacher_structured
hidden: 2048
layers: 18
heads: 16
intermediate: 6144
training: reflow + consistency
supervision: 4-step
engine shape: s3328
```

## 4. FLEX AE28과 현재 baseline의 차이

FLEX 실험 경로에는 별도 AE28 Action Expert가 있다.

```text
FLEX AE28 ONNX:
/workspace/models/student_weights/flex_k512_fp16/ae28/ae28_single_step.onnx

FLEX AE28 TensorRT plan:
/workspace/models/student_weights/engines/flex_k512_fp16/ae28/ae28_single_step.plan
```

AE28 contract:

```text
inputs:
  noisy_action [B, 64, 2]
  timestep     [B, 1, 1]
  position_ids [3, B, 64]
  past_keys    [28, B, 8, kv_seq_len, 128]
  past_values  [28, B, 8, kv_seq_len, 128]

output:
  velocity [B, 64, 2]
```

하지만 현재 실차 후보 baseline은 FLEX가 아니라 legacy `alpa1.5 + visual_fp8_rebuild + flowmatching_20260424_trt_fp8` 조합이다. 따라서 지금 개선하려는 대상은 우선 FLEX AE28이 아니라 legacy FM/Action Expert student 쪽이다.

## 5. 현재 문제 정의

현재 모델은 직진 및 완만한 곡률 구간에서는 상대적으로 안정적이다. 하지만 실차 주행에서 가장 걱정되는 failure mode는 아래와 같다.

```text
1. 좌회전 진입 지점이 늦음
2. 좌회전 경로가 GT보다 바깥으로 크게 돎
3. 직진/좌회전/우회전이 모두 가능한 분기점에서 route intent를 확실히 고르지 못함
4. 신호, 노면 화살표, 명확한 lane cue가 없는 교차로에서 smooth/late turn을 선호함
```

이 문제는 pure pursuit LD, latency compensation으로 일부 완화할 수 있지만, 모델이 생성하는 path 자체가 늦게 꺾이거나 바깥으로 도는 경우에는 AE/FM 재학습 또는 distillation이 필요하다.

## 6. 제안하는 학습 방향

### Option A. AE/FM hard-case fine-tuning

가장 먼저 추천하는 방법이다.

목표:

```text
현재 legacy VLM/visual backbone은 고정
post-VLM Action Expert / Flow Matching student만 fine-tune
좌회전 진입 타이밍, curvature, wide-turn failure를 줄임
```

학습 target:

```text
GT future trajectory
GT에서 계산한 heading / curvature / progress
offline legacy 실패 케이스의 corrected trajectory
```

강화해야 할 데이터:

```text
신호 없는 좌회전
노면 화살표 없는 좌회전
직진/좌회전 모두 가능한 분기점
직진/우회전 모두 가능한 분기점
완만한 커브와 실제 회전이 헷갈리는 구간
실차 no-nav 주행에서 바깥으로 돈 구간
```

권장 batch 구성:

```text
50% normal straight / mild curve
30% branch / left-right turn
20% failure replay / hard case
```

### Option B. Official 10B teacher distillation

공식 10B는 실시간 배포에는 느리지만 teacher로는 유용하다.

목표:

```text
official 10B의 AE/FM path 또는 128 discrete-token decoded path를 teacher로 사용
현재 legacy fast AE/FM student가 teacher path를 모방
실차 배포 runtime은 현재처럼 빠른 legacy engine 유지
```

Teacher target 선택 기준:

```text
teacher가 GT보다 좋거나 비슷하면 teacher 사용
teacher가 GT와 크게 다르면 GT 우선
분기점에서는 route intent와 실제 GT route가 일치하는 teacher만 사용
```

Distillation loss:

```text
L_teacher_xy
L_teacher_heading
L_teacher_curvature
L_teacher_progress
L_gt_xy for available GT
```

### Option C. Nav-conditioned AE/FM fine-tuning

분기점에서 목적 경로를 확실히 고르려면 장기적으로 필요하다.

목표:

```text
runtime CFG로 w30처럼 강하게 미는 방식이 아니라
학습 입력에 route intent를 넣어 single-pass로 자연스럽게 반영
```

Route intent examples:

```text
none
go straight
turn left ahead
turn right ahead
follow lane
```

주의:

```text
nav가 없을 때도 기본 주행이 안정적이어야 함
nav는 강제 조향 신호가 아니라 ambiguity 해소 조건이어야 함
runtime latency가 늘어나는 dual-cache CFG는 실차 기본 경로로 쓰지 않는 것이 좋음
```

## 7. Loss 제안

단순 ADE만 최적화하면 직진 안정성은 좋아 보여도 좌회전 진입 타이밍이 늦거나, 반대로 hard-case만 맞추다가 직진이 흔들릴 수 있다. 주행 가능성을 같이 반영해야 한다.

권장 loss:

```text
L_xy:
  trajectory position loss. 0-3초 구간에 높은 weight.

L_heading:
  진행 방향 오차. 좌회전 진입부에서 특히 중요.

L_curvature:
  GT 또는 teacher curvature와의 오차. late turn / wide turn 완화 목적.

L_curvature_smooth:
  curvature jerk, 경로 불연속 억제.

L_progress:
  너무 적게 가거나 너무 많이 가는 경로 억제.

L_lateral_branch:
  분기점에서 선택된 branch 방향으로 lateral movement를 맞춤.

L_reverse:
  후진성 또는 비정상 뒤로 가는 point 강한 penalty.
```

Hard-case weighting:

```text
좌회전 진입 0-3초: weight 높게
분기점 branch direction: weight 높게
straight 안정성: regression 방지용 normal batch 유지
```

## 8. 평가 기준

현재 baseline:

```text
2026-06-12-test1 latency-aware stitched ADE: 0.17 m
2026-06-12-test1 p90: 0.40 m
2026-06-12-test1 max: 1.37 m

2026-06-12-test2 latency-aware stitched ADE: 0.19 m
2026-06-12-test2 p90: 0.42 m
2026-06-12-test2 max: 1.42 m
```

새 모델 acceptance criteria:

```text
1. straight / mild curve에서 기존 seed23 대비 regression 없어야 함
2. left-turn hard-case에서 wide-turn max lateral error 감소
3. branch 구간에서 GT route 또는 nav route 선택률 개선
4. curvature jerk 증가 금지
5. latency는 기존 FM plan 대비 크게 증가하면 안 됨
6. TensorRT engine contract는 가능하면 유지
```

평가 리포트 필수 항목:

```text
overall ADE / FDE / p90 / max
hard-left subset ADE / FDE / max lateral error
branch subset route-choice accuracy
curvature jerk
latency-aware stitched rollout
GT vs old legacy vs new student overlay PNG
failure top-10
```

## 9. 외부 서버에 요청할 항목

현재 서버에는 TensorRT plan과 ONNX source는 있으나, 학습 가능한 PyTorch checkpoint는 명확히 보이지 않는다. 외부 학습 서버에서 아래를 찾아야 한다.

필수:

```text
1. current legacy FM/AE student PyTorch checkpoint
2. training config
3. model class / training code
4. dataset loader
5. ONNX export script
6. TensorRT build script
```

찾을 checkpoint 후보 이름:

```text
flow_matching_standard_student_reflow_consistency_4step_full_ft_e12
teacher_structured_student_reflow_consistency_step4_mlp6144
student_reflow_consistency_step4
reflow_consistency_step4
teacher_structured
```

최종 산출물:

```text
new_student_checkpoint.pt
new_student.onnx
new_student.onnx.data
new_student.constants.json
new_student_thor_fp16.plan
new_student_thor_fp8.plan
eval_report.md
overlay_pngs/
```

## 10. 외부 서버 전달용 제안 멘트

아래 내용을 그대로 학습 서버 담당자에게 전달하면 된다.

```text
현재 실차 후보는 legacy Alpamayo VLM + prefill-KV FM + seed23 no-nav 조합입니다.
이 조합은 직진/완만한 커브에서는 latency와 ADE가 가장 안정적이지만,
좌회전 진입이 늦고 분기점에서 경로가 바깥으로 크게 도는 failure가 남아 있습니다.

현재 배포 engine은 다음 TensorRT plan입니다.
/workspace/models/alpamayo_runtime/fm/flowmatching_20260424_trt_fp8/engine_thor/teacher_structured_student_reflow_consistency_step4_mlp6144_one_step_fp8_s3328_thor.plan

이 plan은 학습할 수 없으므로, 엔진으로 내리기 전 PyTorch checkpoint를 찾아서
Action Expert / Flow Matching student를 fine-tune 또는 distillation하고 싶습니다.

우선순위는 VLM/ViT/LLM 전체 재학습이 아니라 post-VLM AE/FM student 재학습입니다.
학습 목표는 straight 안정성은 유지하면서 좌회전/분기 hard-case에서
late turn, wide turn, branch ambiguity를 줄이는 것입니다.

추천 학습:
1. 좌회전/분기 hard-case weighted fine-tuning
2. official 10B output과 GT를 teacher로 쓰는 trajectory distillation
3. 필요하면 nav-conditioned AE/FM fine-tuning

평가는 기존 seed23 no-nav baseline과 비교합니다.
straight/mild curve regression이 없어야 하고,
left-turn hard-case max lateral error와 branch route-choice failure가 줄어야 합니다.
최종 산출물은 PyTorch checkpoint, ONNX, Thor TensorRT plan, 평가 리포트, overlay PNG입니다.
```

## 11. 결론

현재 성능 개선을 위해 가장 현실적인 재학습 대상은 VLM/LLM 전체가 아니라 post-VLM Action Expert / Flow Matching student이다. 현재 plan 이름에 포함된 `teacher_structured_student_reflow_consistency_step4_mlp6144` 계열의 PyTorch checkpoint를 찾아서, hard-case fine-tuning 또는 10B teacher distillation을 진행하는 것이 좋다.
