# Current Model Distillation Deployment Plan

## 목적

현재 실차 후보는 `legacy Alpamayo VLM + prefill-KV FM + seed 23` 조합이다. 이 조합은 지금까지 확인한 범위에서 ADE와 latency가 가장 안정적이다.

목표는 공식 10B 모델을 실시간으로 직접 쓰는 것이 아니라, 10B와 GT/hard-case 정보를 teacher로 사용해서 현재 빠른 legacy runtime의 경로 품질을 올리는 것이다. 배포 타깃은 계속 Jetson Thor의 TensorRT-Edge-LLM runtime이며, latency를 크게 늘리지 않는 것이 핵심 조건이다.

## 현재 기준선

기준 모델:

```text
LLM engine:
/workspace/models/alpamayo_runtime/engines/alpa1.5

Multimodal engine:
/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild

FM engine:
/workspace/models/alpamayo_runtime/fm/flowmatching_20260424_trt_fp8/engine_thor/teacher_structured_student_reflow_consistency_step4_mlp6144_one_step_fp8_s3328_thor.plan
```

기준 inference 설정:

```text
nav: off
diffusion_seed: 23
diffusion_num_steps: 2
alpamayo_fm_use_prefill_kv: on
```

확인된 latency-aware rollout 결과:

```text
2026-06-12-test1: ADE 0.17 m, p90 0.40 m, max 1.37 m, FDE 0.29 m, mean latency 989 ms
2026-06-12-test2: ADE 0.19 m, p90 0.42 m, max 1.42 m, FDE 0.78 m, mean latency 980 ms
```

관련 산출물:

```text
/workspace/alpamayo_vlm/output/legacy_prefill_seed_sweep_20260612/seed_sweep_report.md
/workspace/alpamayo_vlm/output/legacy_seed23_latency_rollout_20260612/latency_rollout_report.md
```

## 왜 10B 직접 배포가 아니라 distillation인가

공식 10B 모델은 경로 품질 검증과 teacher 생성에는 유용하지만, 실차 제어 루프에 직접 넣기에는 latency 부담이 크다. 현재 실차 목표는 약 10 km/h, pure pursuit LD 1.5-2.0 m 수준이므로, 경로 품질뿐 아니라 publish delay와 frame freshness가 중요하다.

따라서 권장 방향은 다음과 같다.

```text
공식 10B / GT / hard-case annotations -> teacher target 생성
현재 legacy VLM prefill-KV representation -> student AE/FM 추가학습
Jetson Thor TensorRT plan으로 export -> 기존 planner_live command에서 FM engine만 교체
```

## 개선하고 싶은 문제

현재 모델은 직진 및 완만한 커브에서는 안정적이지만, 다음 상황에서 약점이 있다.

```text
1. 신호나 노면 화살표가 뚜렷하지 않은 좌회전
2. 직진과 좌/우회전이 모두 가능한 분기점
3. 좌회전 시작 시점이 늦어져 경로가 바깥으로 크게 도는 케이스
4. vision만으로 route intent가 애매한 케이스
```

nav-guidance weight를 강하게 주는 방식은 일부 구간에서 intent를 밀어줄 수 있으나, w30 실험에서는 좌회전 후반부가 GT보다 바깥으로 벌어지고 latency도 증가했다. 따라서 runtime CFG를 강하게 쓰는 방식보다, 학습 단계에서 branch/turn intent와 hard-case trajectory를 반영하는 쪽이 우선이다.

## 권장 distillation 단계

### Phase 1. Output-level trajectory distillation

가장 먼저 할 단계다. 현재 student의 구조는 크게 바꾸지 않고, teacher/GT trajectory를 target으로 AE/FM 출력 경로를 맞춘다.

입력:

```text
4-camera images: left, front, right, front_tele
ego history: 기존 request bank의 ego_history_xyz_npy, ego_history_rot_npy
VLM prefill KV or post-VLM hidden representation
optional route intent: none / turn left / turn right / go straight
```

Target:

```text
GT future trajectory
official 10B AE/FM output trajectory
official 10B 128 discrete-token decoded trajectory
legacy model output for stable straight cases
```

권장 target mixing:

```text
straight/normal curve: GT 중심, legacy 안정 경로도 참고
branch/turn hard case: GT + official 10B 중 주행 품질이 좋은 경로를 teacher로 선택
teacher와 GT가 크게 다르면 GT 우선, teacher는 soft target으로만 사용
```

### Phase 2. Hard-case weighted fine-tuning

좌회전, 분기점, 큰 곡률 구간에 sampling weight를 높인다. 단순 ADE 평균만 낮추면 직진 안정성이 깨질 수 있으므로 hard-case와 normal-case batch 비율을 분리한다.

권장 batch 구성:

```text
50-60% normal straight / mild curve
25-35% left/right branch and turn
10-15% failure replay cases from 실차 or offline rollout
```

### Phase 3. Nav-conditioned fine-tuning

실차에서 최종적으로 route intent를 넣고 싶다면 runtime CFG가 아니라 학습 입력으로 nav intent를 넣는 쪽이 좋다.

권장 route intent:

```text
go straight
turn left ahead
turn right ahead
follow lane
```

중요한 점:

```text
nav가 없어도 기본 주행이 안정적이어야 한다.
nav는 경로를 강제로 꺾는 신호가 아니라 분기점 ambiguity를 줄이는 약한 condition이어야 한다.
runtime에서는 nav-CFG dual pass 없이 single-pass conditioned model로 가는 것이 latency 측면에서 유리하다.
```

## Loss 설계 권장안

ADE만 쓰지 말고 주행 가능한 경로를 선호하는 loss를 같이 둔다.

```text
L_xy:
  horizon별 위치 오차. 0-3초 구간에 더 높은 weight.

L_heading:
  GT/teacher 진행 방향과의 heading error.

L_curvature:
  좌회전/우회전 진입부에서 curvature timing을 맞추는 항.

L_smooth:
  과한 curvature jerk와 불연속 경로를 억제.

L_progress:
  너무 느리거나 과하게 진행하는 경로를 억제.

L_reverse:
  후진성/비정상 뒤로 가는 포인트 강한 패널티.

L_lateral_branch:
  분기점에서 GT lane 중심 또는 teacher-selected route와 lateral 방향을 맞추는 항.
```

권장 종합 score:

```text
drive_cost =
  ADE
  + 0.5 * FDE
  + heading_error
  + curvature_jerk_penalty
  + reverse_penalty
  + under_over_progress_penalty
  + branch_lateral_penalty
```

Seed sweep에서 사용한 `robust_drive_cost`처럼 mean만 보지 말고 p90 및 bad-case rate를 같이 본다.

## 데이터 준비

우선 사용할 데이터:

```text
/workspace/alpamayo_vlm/data/2026-06-12-test1
/workspace/alpamayo_vlm/data/2026-06-12-test2
```

추가로 모아야 하는 데이터:

```text
좌회전 진입이 애매한 분기점
직진/좌회전/우회전이 모두 가능한 교차로
신호와 노면 화살표가 없는 회전 구간
완만한 커브와 실제 회전이 헷갈리는 구간
실차 no-nav 주행 중 모델이 바깥으로 돈 구간
```

Request bank 생성/검증은 기존 스크립트 흐름을 유지한다.

```text
scripts/build_live_chunk_request_bank.py
scripts/run_legacy_seed23_latency_rollout.py
scripts/run_flex_legacy_threeway_dataset_compare.py
```

## Teacher 생성

외부 서버에서는 공식 10B 모델을 offline teacher로 사용한다. latency는 중요하지 않고, 각 sample에 대해 다음 artifact를 저장하는 것이 중요하다.

필수 teacher artifact:

```text
teacher_final_path.json
teacher_ac_decoded_path.json
teacher_128_token_path.json
teacher_timing.json
request_metadata.json
camera_frame_ids.json
ego_history.npy or path references
```

teacher artifact에는 최소한 다음 필드가 있어야 한다.

```json
{
  "dataset": "2026-06-12-test1",
  "sample_id": 505,
  "t0_utc_ns": 1781250615516375296,
  "route_intent": "none or turn left ahead",
  "source": "official_10b_ae_fm",
  "pred_xyz": [[0.0, 0.0, 0.0]],
  "pred_yaw_rad": [],
  "pred_v_mps": [],
  "pred_curvature": [],
  "plan_dt_s": 0.1
}
```

## 외부 서버 배포 패키지 구조

외부 서버에 넘길 distillation 패키지는 아래 구조를 권장한다.

```text
alpamayo_current_model_distill_YYYYMMDD/
  README.md
  configs/
    distill_legacy_seed23.yaml
    dataset_20260612.yaml
  data_index/
    train_samples.jsonl
    val_samples.jsonl
    hard_cases.jsonl
  teacher_artifacts/
    official_10b/
      2026-06-12-test1/
      2026-06-12-test2/
  student_init/
    current_fm_checkpoint_or_onnx/
  training/
    logs/
    checkpoints/
  export/
    fm_student.onnx
    fm_student_fp16.plan
    fm_student_fp8.plan
    DEPLOYMENT_NOTES.md
  eval/
    metrics.json
    latency_rollout_report.md
    overlays/
```

## Export 산출물 요구사항

Jetson Thor 배포를 위해 최종적으로 아래 파일을 받아야 한다.

```text
1. 학습 checkpoint
2. ONNX export
3. TensorRT plan for Thor
4. TensorRT build log
5. DEPLOYMENT_NOTES.md
6. validation metrics
7. representative path overlay PNGs
```

가능하면 기존 runtime interface를 유지한다.

```text
LLM engine dir: unchanged
Multimodal engine dir: unchanged
FM engine: replace only this plan
```

이 구조가 유지되면 planner command에서 `--fm-engine`만 교체해서 테스트할 수 있다.

## Thor 검증 명령 예시

새 FM plan이 들어왔을 때 먼저 offline rollout을 확인한다.

```bash
python3 scripts/run_legacy_seed23_latency_rollout.py \
  --dataset-roots /workspace/alpamayo_vlm/data/2026-06-12-test1 /workspace/alpamayo_vlm/data/2026-06-12-test2 \
  --work-root /workspace/alpamayo_vlm/output/distilled_fm_seed23_latency_rollout_YYYYMMDD \
  --engine-dir /workspace/models/alpamayo_runtime/engines/alpa1.5 \
  --multimodal-engine-dir /workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild \
  --fm-engine /workspace/models/alpamayo_runtime/fm/distilled_YYYYMMDD/engine_thor/fm_student.plan \
  --seed 23 \
  --diffusion-num-steps 2 \
  --rerun
```

실차 후보 command는 nav 없이 seed23을 유지한다.

```bash
python3 -m planner_live.planner_live_service \
  --server-url http://127.0.0.1:18080 \
  --engine-dir /workspace/models/alpamayo_runtime/engines/alpa1.5 \
  --multimodal-engine-dir /workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild \
  --fm-engine /workspace/models/alpamayo_runtime/fm/distilled_YYYYMMDD/engine_thor/fm_student.plan \
  --diffusion-seed 23 \
  --diffusion-num-steps 2 \
  --alpamayo-fm-use-prefill-kv \
  --enable-udp-bridge \
  --udp-host 10.179.113.253 \
  --udp-port 5005 \
  --udp-payload-mode text_json \
  --udp-full-plan \
  --udp-send-mode on_result \
  --enable-udp-opencv-ui
```

## Acceptance Criteria

배포 후보로 보려면 최소한 아래 조건을 만족해야 한다.

```text
1. 2026-06-12-test1/test2 straight 및 mild curve에서 기존 seed23 대비 regression 없음
2. latency-aware stitched rollout ADE/p90/max가 기존과 같거나 개선
3. 좌회전 hard-case에서 바깥으로 크게 도는 max lateral error 감소
4. pure pursuit 관점에서 curvature jerk와 steering command 변화량 증가 없음
5. mean latency가 기존 대비 크게 증가하지 않음
6. runtime command에서 LLM/multimodal engine 변경 없이 FM engine 교체만으로 동작
```

현재 기준선보다 나빠지면 reject한다.

```text
mean latency: 기존 대비 +10% 초과 시 reject 후보
straight ADE/p90: 기존 대비 악화 시 reject 후보
branch hard-case: 개선 없으면 distillation 목적 미달
path continuity: stitched path에 불연속 또는 과한 바깥 돌기 있으면 reject 후보
```

## 평가 리포트에 반드시 포함할 항목

```text
1. dataset별 ADE, FDE, p50/p90/max error
2. latency mean/p50/p90
3. hard-case subset score
4. left/right branch subset score
5. pure pursuit rollout 또는 stitched rollout PNG
6. GT vs old legacy vs distilled overlay PNG
7. failure cases top 10
8. seed sensitivity: seed 23 기준, 필요하면 11/17 비교
```

## 리스크와 주의점

1. Teacher가 항상 GT보다 좋은 것은 아니다. 10B output이 GT와 크게 다르면 GT를 우선한다.
2. 좌회전 hard-case만 과하게 학습하면 직진에서 흔들릴 수 있다.
3. nav-CFG weight를 runtime에서 크게 쓰는 방식은 latency와 경로 튐 문제가 있다.
4. offline stitched rollout은 dataset GNSS 기준 anchor를 사용하므로 완전한 closed-loop 주행은 아니다.
5. 실차 검증 시에는 모델 경로, 차량 실제 궤적, steering, speed, LD, request/response timestamp를 반드시 함께 저장해야 한다.

## 권장 진행 순서

```text
1. 기존 legacy seed23 no-nav 기준으로 실차 straight/mild curve 로그 수집
2. 2026-06-12-test1/test2 및 실차 hard-case에서 teacher artifact 생성
3. output-level AE/FM distillation 1차 학습
4. Thor TensorRT plan export
5. offline latency-aware stitched rollout 재검증
6. 실차 straight -> mild curve -> branch 순서로 검증
7. branch 문제가 남으면 nav-conditioned fine-tuning으로 확장
```

## 최종 추천

현재는 `legacy + seed23 + no-nav`가 가장 좋은 실차 baseline이다. 성능 개선은 공식 10B를 실시간으로 직접 내리는 방향보다, 10B/GT를 teacher로 사용해서 현재 빠른 FM/AE student를 추가학습하는 방향이 더 현실적이다.

첫 distillation 목표는 모든 상황에서 큰 개선을 노리기보다, straight 안정성을 유지하면서 좌회전/분기점의 late turn 및 wide turn failure를 줄이는 것으로 잡는다.
