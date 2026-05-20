# Live Dataset Alpamayo Integration Summary

## 개요

본 문서는 라이브 취득 데이터 기반 Alpamayo VLM 모델 실행, 결과 검증, UDP 경로 전달, 제어팀 연동 실험까지의 작업 내용을 정리한다.

작업 범위는 인지 모델 자체의 학습이나 알고리즘 개발보다는, 취득 데이터가 실제 모델 입력으로 들어가고 모델 출력 경로가 제어팀에서 사용할 수 있는 형태로 전달되도록 하는 전체 실행 파이프라인 구축과 검증에 초점을 둔다. 즉 데이터 취득 이후의 포맷 변환, 모델 실행, 결과 시각화, UDP 인터페이스 구성, 제어팀과의 경로 추종 방식 논의 및 실험 운영을 담당했다.

## 사용 데이터 및 모델

사용 데이터:

```text
/workspace/alpamayo_vlm/data/2025-03-31-test2/2026-04-17-test2
```

사용 모델 및 엔진:

```text
/workspace/models/alpamayo_runtime/engines/alpa1.5
/workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild
/workspace/models/alpamayo_runtime/fm/alpamayo15_fm_one_step_mxfp8_thor.plan
```

주요 출력 루트:

```text
/workspace/alpamayo_vlm/output/live_chunk_udp_replay_2026_04_17_test2
/workspace/alpamayo_vlm/output/live_chunk_udp_precompute_1s_2026_04_17_test2
```

## 작업 내용

### 1. 라이브 취득 데이터 구조 파악 및 입력 변환

라이브하게 취득된 raw dataset을 Alpamayo 모델이 처리할 수 있는 request bank 형태로 변환했다. 각 chunk에 대해 카메라 프레임, ego history, GNSS/INS 기반 시각 정보, trajectory 관련 메타데이터를 모델 입력 JSON으로 구성했다.

관련 스크립트:

```text
scripts/build_live_chunk_request_bank.py
```

생성 산출물:

```text
chunkXXXX/request_bank/manifest.json
chunkXXXX/request_bank/summary.json
chunkXXXX/request_bank/requests/
chunkXXXX/request_bank/images/
chunkXXXX/request_bank/ego/
```

이 과정에서 0.1초 단위 sample을 기준으로 모델 입력을 만들고, chunk별로 어떤 sample이 어떤 image/history와 연결되는지 추적 가능하게 구성했다.

### 2. Alpamayo 모델 실행 파이프라인 구성

`llm_inference` persistent server를 사용해 chunk 단위 입력을 Alpamayo VLM 및 post-VLM runtime에 전달하고, 모델 출력 JSON을 저장하는 흐름을 구성했다.

기본 실행 스크립트:

```text
scripts/run_live_chunk_udp_replay.py
```

대표 실행 형태:

```bash
python3 scripts/run_live_chunk_udp_replay.py \
  --dataset-root /workspace/alpamayo_vlm/data/2025-03-31-test2/2026-04-17-test2 \
  --chunk-id 4 \
  --work-root /workspace/alpamayo_vlm/output/live_chunk_udp_replay_2026_04_17_test2 \
  --replay-mode latest_only \
  --engine-dir /workspace/models/alpamayo_runtime/engines/alpa1.5 \
  --multimodal-engine-dir /workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild \
  --fm-engine /workspace/models/alpamayo_runtime/fm/alpamayo15_fm_one_step_mxfp8_thor.plan
```

지원한 실행 모드:

```text
sequential: 선택된 request를 순서대로 모두 처리
latest_only: 실제 live 상황처럼 inference 중 stale frame을 skip하고 최신 frame 기준으로 처리
```

`latest_only`는 모델 추론 시간이 실제 0.1초 frame interval보다 길 때, live 시스템에서 어떤 입력이 실제 처리되는지 재현하기 위한 모드로 사용했다.

### 3. 모델 결과 후처리 및 경로 artifact 생성

모델 출력 JSON에서 최종 경로를 해석하고, 제어팀에서 사용할 수 있는 path payload와 시각화용 artifact를 생성했다.

생성한 주요 artifact:

```text
latest_final_path.json
latest_ac_decoded_path.json
latest_gt_path.json
latest_udp_payload.json
latest_tx_row.json
tx_log.json
summary.json
```

경로 source는 다음 중 선택 가능하게 구성했다.

```text
ac_decoded
final
gt
```

실험에서는 기존 제어 경로 흐름과 맞추기 위해 기본적으로 `ac_decoded`를 사용했다.

### 4. HTML 기반 결과 확인 도구 유지 및 확장

이전에 사용하던 HTML viewer 형식을 유지하면서 새 데이터셋 결과를 확인할 수 있게 했다. chunk별 viewer는 각 chunk artifact를 기준으로 동작하고, 최신 이미지, 모델 출력 path, UDP payload, tx log 상태를 확인할 수 있다.

또한 chunk0000부터 chunk0009까지 전체 chunk를 한 화면에서 확인할 수 있는 all-chunks viewer를 구성했다.

전체 viewer:

```text
http://127.0.0.1:8800/viewer
```

확인한 chunk:

```text
chunk0000 ~ chunk0009
```

처음에는 일부 chunk artifact가 없어 missing으로 표시되었으나, chunk0004부터 chunk0009까지 추가로 모델을 실행해 전체 chunk가 viewer에서 확인 가능하도록 만들었다.

### 5. UDP 경로 전달 로직 구현 및 제어팀 연동

모델 결과 경로를 제어팀에서 받을 수 있도록 UDP payload 형태로 전송했다. 전송 대상은 실험 중 다음 포트를 사용했다.

```text
192.168.0.29:5005
```

기존 path payload에 추론 시간을 포함하도록 `inference_time_s` 필드를 추가했다.

```json
{
  "inference_time_s": 2.7
}
```

해당 값은 float 타입이며 초 단위로 기록된다. payload뿐 아니라 tx log에도 함께 저장되도록 했다.

기록 위치:

```text
latest_udp_payload.json
latest_tx_row.json
tx_log.json
```

이를 통해 제어팀은 수신한 path가 어떤 sample에서 생성되었는지, 모델 추론 시간이 얼마나 걸렸는지, 어떤 replay mode로 전송되었는지를 같이 확인할 수 있다.

### 6. Precompute 후 fixed-interval UDP replay 구현

실시간 inference 속도에 따라 frame이 skip되는 `latest_only` 방식과 별개로, 모든 입력 frame에 대한 path를 먼저 계산한 뒤 일정 주기로 UDP 전송하는 precompute replay 방식을 새로 구현했다.

신규 스크립트:

```text
scripts/run_live_chunk_udp_precompute_replay.py
```

이 방식의 목적:

```text
1. 모든 선택 frame에 대해 모델 추론 수행
2. path payload를 전부 저장
3. 모델 프로세스 종료
4. 저장된 path를 고정 주기로 UDP 전송
```

0.1초 간격 전체 replay 예시:

```bash
python3 scripts/run_live_chunk_udp_precompute_replay.py \
  --dataset-root /workspace/alpamayo_vlm/data/2025-03-31-test2/2026-04-17-test2 \
  --chunk-id 0 \
  --work-root /workspace/alpamayo_vlm/output/live_chunk_udp_replay_2026_04_17_test2 \
  --engine-dir /workspace/models/alpamayo_runtime/engines/alpa1.5 \
  --multimodal-engine-dir /workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild \
  --fm-engine /workspace/models/alpamayo_runtime/fm/alpamayo15_fm_one_step_mxfp8_thor.plan \
  --udp-host 192.168.0.29 \
  --udp-port 5005 \
  --send-interval-s 0.1
```

실험 결과:

```text
chunk0000 selected_request_count: 597
precomputed_request_count: 597
udp_sent_count: 597
send_interval_s: 0.1
총 전송 시간: 약 59.6초
```

### 7. 저장된 path의 UDP 재전송 기능 추가

모델 재계산 없이, 이미 저장된 precomputed payload만 다시 UDP로 전송할 수 있는 `--send-only` 모드를 추가했다.

재전송 명령 예시:

```bash
python3 scripts/run_live_chunk_udp_precompute_replay.py \
  --dataset-root /workspace/alpamayo_vlm/data/2025-03-31-test2/2026-04-17-test2 \
  --chunk-id 0 \
  --work-root /workspace/alpamayo_vlm/output/live_chunk_udp_replay_2026_04_17_test2 \
  --send-only \
  --udp-host 192.168.0.29 \
  --udp-port 5005 \
  --send-interval-s 0.1 \
  --disable-viewer
```

이 기능을 통해 제어팀이 같은 path sequence를 여러 번 반복 수신하면서 경로 추종 로직을 검증할 수 있게 했다.

### 8. 1초 간격 추론 데이터셋 생성

0.1초 간격 전체 path 외에, 1초 간격으로 샘플링한 path set도 별도로 생성했다. 기존 0.1초 request bank에서 `request_stride=10`을 적용해 sample_id가 0, 10, 20, ... 형태가 되도록 구성했다.

실행 명령:

```bash
python3 scripts/run_live_chunk_udp_precompute_replay.py \
  --dataset-root /workspace/alpamayo_vlm/data/2025-03-31-test2/2026-04-17-test2 \
  --chunk-id 0 \
  --work-root /workspace/alpamayo_vlm/output/live_chunk_udp_precompute_1s_2026_04_17_test2 \
  --request-bank-root /workspace/alpamayo_vlm/output/live_chunk_udp_replay_2026_04_17_test2/chunk0000/request_bank \
  --request-stride 10 \
  --precompute-only \
  --disable-viewer \
  --engine-dir /workspace/models/alpamayo_runtime/engines/alpa1.5 \
  --multimodal-engine-dir /workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild \
  --fm-engine /workspace/models/alpamayo_runtime/fm/alpamayo15_fm_one_step_mxfp8_thor.plan \
  --send-interval-s 1.0
```

생성 결과:

```text
precomputed_request_count: 60
udp_sent_count: 0
sample_id: 0, 10, 20, ... 590
actual_offset step: 1.0s
```

저장 위치:

```text
/workspace/alpamayo_vlm/output/live_chunk_udp_precompute_1s_2026_04_17_test2/chunk0000/precompute_udp_replay/artifacts/precomputed_udp_payloads.jsonl
/workspace/alpamayo_vlm/output/live_chunk_udp_precompute_1s_2026_04_17_test2/chunk0000/precompute_udp_replay/artifacts/tx_log.json
```

1초 간격 path 전송 명령:

```bash
python3 scripts/run_live_chunk_udp_precompute_replay.py \
  --dataset-root /workspace/alpamayo_vlm/data/2025-03-31-test2/2026-04-17-test2 \
  --chunk-id 0 \
  --work-root /workspace/alpamayo_vlm/output/live_chunk_udp_precompute_1s_2026_04_17_test2 \
  --send-only \
  --udp-host 192.168.0.29 \
  --udp-port 5005 \
  --send-interval-s 1.0 \
  --disable-viewer
```

## 제어팀 협업 내용

제어팀과의 연동을 위해 모델 출력 path를 제어 모듈이 수신 가능한 UDP JSON payload로 전달했다. 또한 실시간 추론 방식과 precompute replay 방식의 차이를 구분해 제공했다.

협의한 주요 포인트:

```text
실시간 모드에서는 inference latency 때문에 중간 frame이 skip될 수 있음
precompute replay에서는 모든 계산 결과를 고정 주기로 전달할 수 있음
제어팀은 동일 path sequence를 반복 수신하며 추종 로직을 검증할 수 있음
payload에 inference_time_s를 포함해 모델 지연 시간을 함께 고려할 수 있음
0.1초 간격과 1초 간격 데이터를 모두 제공해 제어 입력 주기별 비교가 가능함
```

이 작업을 통해 인지 결과가 제어팀의 경로 추종 실험으로 연결되는 end-to-end 흐름을 구성하고 검증했다.

## 주요 산출물

코드:

```text
scripts/run_live_chunk_udp_replay.py
scripts/run_live_chunk_udp_precompute_replay.py
```

데이터 변환 및 request bank:

```text
/workspace/alpamayo_vlm/output/live_chunk_udp_replay_2026_04_17_test2/chunkXXXX/request_bank
```

0.1초 precompute 결과:

```text
/workspace/alpamayo_vlm/output/live_chunk_udp_replay_2026_04_17_test2/chunk0000/precompute_udp_replay
```

1초 precompute 결과:

```text
/workspace/alpamayo_vlm/output/live_chunk_udp_precompute_1s_2026_04_17_test2/chunk0000/precompute_udp_replay
```

HTML viewer:

```text
http://127.0.0.1:8800/viewer
```

UDP 전송 대상:

```text
192.168.0.29:5005
```

## 기여 범위 요약

본 작업의 기여 범위는 다음과 같이 정리할 수 있다.

```text
라이브 취득 데이터의 모델 입력 포맷 변환
Alpamayo VLM 모델 실행 파이프라인 구성
chunk 단위 모델 결과 생성 및 저장
HTML 기반 결과 시각화 및 전체 chunk viewer 구성
모델 출력 path 후처리 및 UDP payload 구성
inference_time_s 필드 추가
제어팀 수신용 UDP 전송 및 반복 재전송 기능 구현
0.1초/1초 간격 path dataset 생성
제어팀과 경로 추종 실험 방식 논의 및 end-to-end 검증 지원
```

요약하면, 인지 모델 자체의 학습이나 모델 내부 알고리즘 개발을 제외하고, 라이브 데이터가 모델 입력으로 들어가고 모델 출력 경로가 제어팀의 경로 추종 실험으로 이어지는 전체 파이프라인을 구성하고 검증하는 역할을 수행했다.
