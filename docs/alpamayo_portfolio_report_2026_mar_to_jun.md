# Alpamayo VLM 실차 주행 파이프라인 구축 및 최적화 리포트

## 개요

2026년 3월 이후 Alpamayo VLM 기반 자율주행 경로 생성 시스템을 실제 차량 테스트에 연결하기 위한 데이터 수집, TensorRT-Edge-LLM 런타임 통합, 모델 추론 최적화, 경로 시각화, 제어기 연동 검증, 모델 성능 평가 작업을 진행했다.

핵심 목표는 단순히 모델을 한 번 실행하는 것이 아니라, 차량에서 수집되는 4-camera, GNSS/INS, ego history 데이터를 모델 입력으로 안정적으로 변환하고, Jetson Thor 환경에서 TensorRT 기반 추론을 수행한 뒤, 결과 경로를 제어 시스템이 사용할 수 있는 형태로 전달하는 end-to-end 파이프라인을 만드는 것이었다.

## 역할

담당 범위:

```text
1. 실차 raw dataset 구조 설계 및 offline replay 입력 변환
2. Alpamayo VLM / TensorRT-Edge-LLM runtime 실행 파이프라인 구성
3. persistent inference, request bank, staging, result parsing 구현 및 검증
4. UDP 기반 경로 전달 및 제어팀 연동
5. 모델 variant별 benchmark와 경로 품질 비교
6. flow matching seed sweep, latency-aware stitched rollout 평가
7. pure pursuit 제어 관점의 lookahead distance, 좌표계, latency 영향 분석
8. nav guidance, FLEX, official 10B, legacy 모델 비교 및 distillation 방향 제안
```

사용 기술:

```text
TensorRT-Edge-LLM
Jetson Thor
C++ / Python
TensorRT plan / engine
VLM multimodal runtime
Flow Matching Action Expert
UDP bridge
GNSS/INS 기반 trajectory 평가
Parquet / NPZ / JSON request pipeline
Matplotlib 기반 경로 시각화
```

## 1. 실차 Raw Dataset 설계

초기에는 모델 실행보다 먼저 실차 데이터를 어떤 형태로 저장하고 재생할지 정의하는 작업이 필요했다. 차량 플랫폼은 4개 카메라, IMU, GNSS/INS를 사용하므로 sensor timestamp를 보존하고 offline에서 재현 가능한 구조가 중요했다.

설계한 raw dataset 요구사항:

```text
camera: 4 streams, 30 Hz
IMU: 100 Hz
GNSS/INS: 10 Hz
time sync: UTC 기준 유지
camera storage: chunked video + frame index parquet
sensor metadata: session_meta.json
calibration: intrinsics / mount placeholder schema
```

주요 판단:

```text
1. raw 단계에서 downsample하지 않고 원본 sensor rate를 유지
2. frame별 PNG/JPG 저장 대신 chunked video와 parquet index 사용
3. 모든 sensor record에 UTC timestamp를 저장
4. calibration이 없어도 나중에 확장 가능한 placeholder schema 정의
```

관련 문서:

```text
ALPAMAYO_RAW_DATA_TASK.md
```

## 2. Planner Container 및 Live Sample Contract 설계

실차 runtime에서는 sensor container와 planner container를 분리하는 구조를 가정했다. planner container는 sensor container가 제공하는 최신 NPZ sample을 받아 validation, staging, inference, result parsing을 담당한다.

정의한 sample contract:

```text
image_frames: [Cam, T, C, H, W]
camera_indices: [Cam]
ego_history_xyz: [1, 1, 16, 3]
ego_history_rot: [1, 1, 16, 3, 3]
relative_timestamps: [Cam, T]
absolute_timestamps: [Cam, T]
t0_us: [1]
fixed_delta_seconds: [1]
clip_id: [1]
camera_order: [Cam]
```

현재 deployment 기준:

```text
camera order: left, front, right, front_tele
camera ids: 0, 1, 2, 6
image shape: 320 x 576 RGB
frame offsets: t0-300ms, t0-200ms, t0-100ms, t0
ego history length: 16
latest-only scheduling: enabled
```

핵심 설계:

```text
1. inference 중 backlog가 쌓이지 않도록 latest-only 정책 적용
2. 현재 runtime boundary는 PNG / NPY / request JSON staging 유지
3. direct-memory API는 장기 목표로 분리
4. sample validation failure 시 inference를 실행하지 않도록 설계
```

관련 문서:

```text
PLANNER_CONTAINER_HANDOFF.md
```

## 3. Offline Dataset을 Alpamayo Request Bank로 변환

실차에서 취득한 dataset을 모델 입력 request bank로 변환하는 파이프라인을 구축했다. chunk별 camera frame, ego history, GNSS/INS trajectory를 정렬해서 모델이 요구하는 JSON request, image staging, ego history NPY 파일을 생성했다.

생성 산출물:

```text
request_bank/manifest.json
request_bank/summary.json
request_bank/requests/
request_bank/images/
request_bank/ego/
```

이 구조를 통해 어떤 모델 출력이 어떤 camera frame, 어떤 GNSS timestamp, 어떤 ego history에서 생성됐는지 추적 가능하게 만들었다.

대표 데이터:

```text
2026-04-03-test2
2026-04-17-test2
2026-05-27-test1
2026-06-12-test1
2026-06-12-test2
```

## 4. TensorRT-Edge-LLM 기반 Alpamayo Runtime 통합

Alpamayo VLM은 TensorRT-Edge-LLM 기반 runtime에서 LLM engine, multimodal visual engine, flow matching action expert engine을 조합해서 실행한다.

대표 runtime 구성:

```text
LLM engine:
/workspace/models/alpamayo_runtime/engines/alpa1.5

Multimodal engine:
/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild

FM engine:
/workspace/models/alpamayo_runtime/fm/flowmatching_20260424_trt_fp8/engine_thor/teacher_structured_student_reflow_consistency_step4_mlp6144_one_step_fp8_s3328_thor.plan
```

대표 실행 옵션:

```text
--diffusion-seed 23
--diffusion-num-steps 2
--alpamayo-fm-use-prefill-kv
--enable-udp-bridge
--udp-payload-mode text_json
--udp-full-plan
--udp-send-mode on_result
```

구현/검증한 runtime 기능:

```text
1. persistent llm_inference server 실행
2. request bank batch 실행
3. VLM output, Action Expert output, final path parsing
4. prefill-KV 기반 FM 실행 경로
5. UDP payload generation
6. OpenCV UI / PNG overlay 시각화
```

## 5. UDP 경로 전달 및 제어팀 연동

모델 출력 경로를 제어 시스템에서 사용할 수 있도록 UDP bridge를 구성했다. 경로는 text_json payload로 전송하고, full plan을 포함할 수 있게 했다.

지원한 replay 방식:

```text
1. sequential replay
2. latest-only replay
3. precompute 후 fixed-interval UDP replay
4. send-only 재전송 모드
```

precompute replay의 목적은 모든 frame에 대해 path를 미리 계산한 뒤, 제어팀이 일정한 주기로 경로 수신을 검증할 수 있게 하는 것이다.

기록한 payload 정보:

```text
latest_udp_payload.json
latest_tx_row.json
tx_log.json
inference_time_s
path source
sample timestamp
```

이 작업을 통해 모델 추론 시간과 제어 경로 수신 시점을 분리해서 확인할 수 있었다.

## 6. 시각화 및 평가 도구 구축

모델 결과를 단순 JSON으로 확인하는 것은 한계가 있어서, camera image 위 path overlay와 GNSS GT 비교 plot을 생성하는 시각화 도구를 만들었다.

생성한 시각화 유형:

```text
1. camera image + predicted path overlay
2. GT vs model path bird-eye plot
3. legacy / FLEX / official 10B multi-model overlay
4. seed별 FM 경로 비교
5. latency-aware stitched rollout
6. nav guidance weight sweep plot
7. path-only enlarged report PNG
```

대표 산출물:

```text
output/flex_legacy_threeway_20260612/
output/legacy_prefill_seed_sweep_20260612/
output/legacy_seed23_latency_rollout_20260612/
output/nav_branch_compare_20260612/
```

## 7. 모델 Variant 비교: Legacy, FLEX, Official 10B

여러 모델 조합을 같은 dataset에서 비교했다.

비교한 구성:

```text
1. 기존 legacy 10B-down 모델 + prefill-KV FM
2. FLEX 기반 student model
3. official 10B 모델의 discrete token path
4. official 10B Action Expert path
5. FLEX 4-step / 10-step FM
6. legacy seed별 FM output
```

주요 확인 내용:

```text
1. FLEX token이 LLM request에 제대로 들어가는지 확인
2. Deepstack / visual token 전달 구조 검증
3. FLEX 4-step과 10-step 결과가 겹치는 원인 분석
4. ONNX 업데이트 후 engine 재빌드 및 재시각화
5. official 10B와 legacy 경로 차이 비교
```

결론:

```text
1. official 10B는 teacher 또는 offline reference로 유용하지만 실시간 배포에는 latency 부담이 큼
2. 현재 실차 기준으로는 legacy + prefill-KV + seed 23 조합이 가장 실용적
3. FLEX는 구조 검증과 실험은 가능했으나, 현재 baseline을 대체할 만큼 안정적이라고 판단하기 어려웠음
```

## 8. Flow Matching Seed Sweep 및 기준 Seed 선정

FM diffusion seed가 output path에 미치는 영향을 평가했다. 단순 ADE만 보지 않고 주행 가능성을 반영한 drive cost를 설계했다.

평가 지표:

```text
ADE
FDE
p90 FDE
bad case rate
progress error
lateral drift
curvature penalty
jerk penalty
reverse penalty
robust_drive_cost
```

Seed sweep 결과:

```text
recommended seed: 23
runner-up: 11, 17
기존 command seed 2와 seed 42보다 seed 23이 robust drive cost 기준 우수
```

대표 결과:

```text
seed 23 robust_drive_cost: 5.886
seed 23 mean ADE: 0.964 m
seed 23 mean FDE: 2.858 m
```

관련 산출물:

```text
output/legacy_prefill_seed_sweep_20260612/seed_sweep_report.md
```

## 9. Latency-Aware Stitched Rollout 평가

실제 차량에서는 모델이 경로를 계산하는 동안 이전 경로를 추종하게 된다. 따라서 각 inference 결과를 GT timestamp에 맞춰 단순 비교하는 것만으로는 실제 주행 품질을 판단하기 어렵다.

이를 위해 모델 latency만큼 plan 앞부분을 건너뛰고, 다음 plan publish 시점까지의 구간을 이어붙이는 stitched rollout 평가를 만들었다.

평가 방식:

```text
1. 약 1초 간격 request 생성
2. 각 request의 total_post_vlm_ms 또는 request_wall_ms를 publish delay로 사용
3. delay만큼 plan 앞부분 skip
4. 다음 plan publish 시점까지 직전 plan segment를 추종한다고 가정
5. 전체 segment를 하나의 continuous path로 stitch
6. GNSS GT와 비교
```

Legacy seed23 no-nav 결과:

```text
2026-06-12-test1:
  samples: 60
  mean latency: 989.3 ms
  ADE: 0.17 m
  p90 error: 0.40 m
  max error: 1.37 m
  FDE: 0.29 m

2026-06-12-test2:
  samples: 60
  mean latency: 980.3 ms
  ADE: 0.19 m
  p90 error: 0.42 m
  max error: 1.42 m
  FDE: 0.78 m
```

관련 산출물:

```text
output/legacy_seed23_latency_rollout_20260612/latency_rollout_report.md
```

## 10. Pure Pursuit 제어 관점 분석

모델 경로를 실제 차량이 추종할 때 pure pursuit controller의 lookahead distance가 중요하다. 경로 품질과 10 km/h 수준의 저속 주행을 기준으로 LD 범위를 검토했다.

권장 초기 설정:

```text
speed: 10 km/h
LD start: 1.5 m
LD upper test: 2.0 m
```

판단:

```text
1. 너무 큰 LD는 좌회전/커브에서 늦게 따라가며 바깥으로 돌 가능성이 있음
2. 너무 작은 LD는 직진에서 조향이 예민해질 수 있음
3. 실차 초기 테스트는 LD 1.5 m에서 시작하고, 흔들림이 있으면 1.8-2.0 m로 조정
```

좌표계 관련 검토:

```text
1. GNSS 기준 경로와 comma/pure pursuit 기준 좌표계 차이 검토
2. GNSS가 후륜축에 있고 comma가 전방 유리에 있을 때 offset 영향 분석
3. pure pursuit이 후륜축 기준이면 실제 제어 기준점과 모델 경로 기준점 정렬이 중요함
```

실차 로그로 반드시 남겨야 할 항목:

```text
model path
vehicle trajectory
steering command
speed
LD
request timestamp
response timestamp
```

## 11. Nav Guidance 실험과 결론

분기점에서 좌회전/직진 의도를 명확히 주기 위해 nav text와 guidance weight를 실험했다.

실험한 방식:

```text
nav text: turn left ahead
guidance weight: 3, 6, 10, 20, 25, 30, 35, 40, 50, 80, 120
target data: 2026-06-12-test1/test2 branch area
```

관찰:

```text
1. test1 hard-left 구간에서는 w30-w40에서 ADE/FDE가 일부 개선
2. test2에서는 w25-w40이 일관되게 개선되지 않음
3. w50 이상은 과하게 경로를 밀거나 불안정해질 수 있음
4. nav-CFG dual pass는 latency를 증가시킴
```

manual nav w30 hybrid rollout 결과:

```text
legacy seed23 no-nav:
  ADE 0.17 m, p90 0.40 m, max 1.37 m

manual left nav w30 hybrid:
  ADE 0.25 m, p90 0.60 m, max 2.67 m
```

결론:

```text
현재 실차 baseline은 no-nav + seed23이 더 적절
강한 runtime nav-CFG는 실차 안정성 측면에서 신중해야 함
장기적으로는 nav-conditioned fine-tuning 또는 distillation이 더 적절
```

## 12. INT4AWQ / W4A16 Prefill 병목 분석

INT4AWQ 모델은 decode 단계에서는 FP16보다 빠른 경향이 있지만, prefill 단계에서는 느린 문제가 있었다. 해당 병목은 단순히 kernel 하나를 교체한다고 해결되기 어려운 구조적 문제로 분석했다.

핵심 원인:

```text
1. W4A16 구조에서 weight는 INT4지만 activation은 FP16/BF16 계열
2. prefill은 sequence length가 길어 GEMM shape가 decode와 다름
3. dequantize / scale 적용 / requantize 또는 layout 변환 overhead가 커질 수 있음
4. 기존 TensorRT-Edge-LLM plugin path가 Alpamayo prefill shape에 최적화되어 있지 않음
5. decode에서는 memory bandwidth 이점이 크지만 prefill에서는 kernel launch/layout/dequant overhead가 상대적으로 두드러짐
```

개선 방향:

```text
1. Alpamayo shape 전용 dense W4A16 prepacked kernel
2. weight packing/layout을 runtime이 아니라 build/export 단계에서 고정
3. prefill에서 dequant/requant roundtrip을 줄이는 fused kernel
4. TensorRT plugin path와 engine build profile 재검토
5. 실제 목표 shape 기준 benchmark로 FP16 baseline 대비 이득 확인
```

한계:

```text
단순 kernel 교체만으로는 FP16 prefill을 항상 이긴다고 보장하기 어렵다.
prefill은 INT4 weight compression 이점보다 activation/layout/dequant overhead가 더 커질 수 있다.
```

## 13. Distillation 방향 정리

현재 실차용으로 가장 좋은 조합은 `legacy + seed23 + no-nav`이다. 그러나 좌회전/분기점에서 late turn 또는 wide turn 문제가 남아 있다.

따라서 성능 개선 방향은 official 10B를 실시간으로 직접 배포하는 것이 아니라, official 10B와 GT를 teacher로 사용해 현재 빠른 legacy student를 개선하는 것이다.

권장 구조:

```text
official 10B / GT / hard cases -> teacher trajectories
current legacy runtime representation -> student AE/FM
new FM/AE checkpoint -> TensorRT plan export
existing LLM/multimodal engine 유지
FM engine만 교체해서 검증
```

중요한 정리:

```text
TensorRT engine / plan은 추론 전용이며 추가학습 불가
distillation에는 학습 가능한 PyTorch checkpoint 또는 training graph가 필요
현재 engine bundle은 배포/검증용이지 학습용이 아님
```

관련 문서:

```text
docs/current_model_distillation_deployment_plan.md
```

## 주요 성과 요약

정량 성과:

```text
1. Legacy prefill-KV seed 23 baseline 선정
2. 2026-06-12-test1 stitched rollout ADE 0.17 m 달성
3. 2026-06-12-test2 stitched rollout ADE 0.19 m 달성
4. 평균 post-VLM latency 약 980-990 ms 수준 확인
5. FM seed sweep에서 seed 23을 robust drive cost 기준 추천 seed로 선정
```

시스템 성과:

```text
1. raw dataset -> request bank -> TensorRT inference -> path artifact -> UDP bridge 파이프라인 구축
2. live sample contract와 planner container architecture 정의
3. latest-only scheduling 기반 live inference 구조 설계
4. precompute replay / send-only replay로 제어팀 디버깅 흐름 지원
5. 모델 output을 camera image, GT trajectory, stitched rollout으로 시각화
```

기술적 판단:

```text
1. 실차 baseline은 legacy + seed23 + no-nav가 가장 적절
2. official 10B는 real-time deploy target보다 offline teacher로 적합
3. strong nav-CFG는 일부 케이스에서 경로를 밀 수 있지만 latency와 instability risk가 있음
4. INT4AWQ prefill 개선은 단순 weight quantization보다 shape-specific kernel/layout 최적화가 필요
5. 실차 주행 평가는 ADE뿐 아니라 latency-aware stitched rollout과 pure pursuit 관점의 smoothness를 같이 봐야 함
```

## 포트폴리오용 한 줄 요약

TensorRT-Edge-LLM 기반 Alpamayo VLM 자율주행 경로 생성 모델을 Jetson Thor 실차 테스트 환경에 연결하기 위해 raw sensor dataset 설계, live inference pipeline, UDP 제어 연동, 모델 variant benchmark, latency-aware trajectory evaluation, seed/nav/quantization 분석을 end-to-end로 수행했다.

## 포트폴리오용 Bullet Version

```text
- Built an end-to-end Alpamayo VLM inference pipeline from raw multi-camera vehicle logs to TensorRT-Edge-LLM runtime outputs.
- Designed a replayable raw dataset format for 4-camera, IMU, and GNSS/INS streams with UTC-based synchronization.
- Implemented request-bank generation, persistent inference execution, path artifact generation, and UDP path bridge for control integration.
- Evaluated legacy, FLEX, and official 10B model variants with GT overlays, timing breakdowns, and path-quality metrics.
- Developed latency-aware stitched rollout evaluation to estimate the path a vehicle would actually follow under model compute delay.
- Selected legacy prefill-KV FM seed 23 as the current best real-car baseline based on ADE, latency, and robust drive-cost metrics.
- Analyzed nav guidance, pure pursuit LD, coordinate-frame offsets, and INT4AWQ prefill bottlenecks for real-car deployment decisions.
- Proposed a distillation roadmap using official 10B and GT trajectories as teachers while preserving the fast legacy runtime as the deployable student.
```

## 향후 개선 방향

```text
1. 실차 straight / mild curve / branch 순서로 no-nav seed23 baseline 검증
2. 좌회전 hard-case와 분기점 데이터를 추가 수집
3. official 10B와 GT를 teacher로 사용하는 student AE/FM distillation 진행
4. 학습 가능한 current student checkpoint 확보
5. distilled FM TensorRT plan export 후 기존 runtime에서 A/B 테스트
6. closed-loop 주행 로그 기반으로 model path와 vehicle trajectory 차이를 재평가
```
