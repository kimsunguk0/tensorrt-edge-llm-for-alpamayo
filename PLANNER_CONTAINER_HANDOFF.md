# Planner Container Handoff

This document is for the next Codex / engineer who will implement the **planner / inference-side container** that receives live samples from the sensor container and runs Alpamayo inference.

## Critical Assumption

Unlike the sensor container, the `planner container` **does contain this TensorRT repo** and is allowed to use and modify it.

That means this document is an implementation guide for the container that will:

- receive planner-ready live samples from the sensor container
- convert those samples into the current TRT runtime boundary
- run persistent `llm_inference`
- publish or hand off planner outputs downstream

## Goal

Build a `planner container` that:

- receives Phase 1 live samples from the sensor container over HTTP `/latest`
- validates the sample contract before inference
- applies a `latest-only` policy so backlog does not grow without bound
- writes the current required PNG / NPY / request JSON staging files
- runs persistent Alpamayo inference
- parses planner outputs into a stable result object for downstream use
- exposes health / freshness state

This container should **not** own:

- real camera ingest
- IMU / GNSS / INS ingest
- camera resize / decode pipeline
- final `50 Hz` control UDP publishing in Phase 1

Its job is to consume synchronized sensor samples and execute the planner runtime reliably.

## Upstream Boundary

Think of the system split like this:

```text
sensor container
  -> /latest NPZ sample
planner container
  -> validates sample
  -> stages runtime files
  -> runs persistent inference
  -> returns plan result
control bridge
  -> consumes plan result
  -> resamples to 50 Hz
  -> sends UDP
```

## Fixed Input Contract From Sensor Container

The planner container must consume the following contract exactly.

### Required Sample Keys

- `image_frames`
- `camera_indices`
- `ego_history_xyz`
- `ego_history_rot`
- `relative_timestamps`
- `absolute_timestamps`
- `t0_us`
- `fixed_delta_seconds`
- `clip_id`
- `camera_order`

### Required Shapes

- `image_frames`: `[Cam, T, C, H, W]`
- `ego_history_xyz`: `[1, 1, 16, 3]`
- `ego_history_rot`: `[1, 1, 16, 3, 3]`
- `camera_indices`: `[Cam]`
- `relative_timestamps`: `[Cam, T]`
- `absolute_timestamps`: `[Cam, T]`
- `t0_us`: `[1]`
- `fixed_delta_seconds`: `[1]`
- `clip_id`: `[1]`
- `camera_order`: `[Cam]`

Axis meaning:

- `image_frames[cam_idx, t_idx]` must correspond to:
  - `camera_indices[cam_idx]`
  - `camera_order[cam_idx]`
- `t_idx = 0,1,2,3` corresponds to the selected frames nearest to:
  - `t0-300 ms`
  - `t0-200 ms`
  - `t0-100 ms`
  - `t0`

### Required Dtypes

- `image_frames`: `uint8`
- `camera_indices`: `int32`
- `ego_history_xyz`: `float32`
- `ego_history_rot`: `float32`
- `relative_timestamps`: `float32`
- `absolute_timestamps`: `int64`
- `t0_us`: `int64`
- `fixed_delta_seconds`: `float32`
- `clip_id`: string array
- `camera_order`: string array

### Fixed Values

- planner camera order: `left`, `front`, `right`, `front_tele`
- planner camera IDs: `0`, `1`, `2`, `6`
- image format: `RGB uint8`
- image tensor shape for the current `4-camera x 4-frame` deployment: `[Cam, T, 3, 320, 576]`
- resize policy on sensor side for the current deployment: bicubic resize to `320 x 576`
- note: raw Qwen single-image resize for `4K (2160 x 3840)` would be `512 x 960`, but that would exceed the current total visual token budget when `16` images are sent in one sample
- current deployment semantics: direct resize from selected raw frame to `320 x 576`, with no crop and no letterbox
- external time basis: `UTC us`
- frame offsets: `t0-300 ms`, `t0-200 ms`, `t0-100 ms`, `t0`
- frame selection: nearest frame to each target timestamp
- `ego_history_xyz`: GNSS/INS fused pose in `t0`-relative ego-local frame
- `ego_history_rot`: INS/Xsens orientation as `3x3` rotation matrix in the same `t0`-relative local frame
- `/latest`: newest valid sample; return `503` if no valid sample exists yet
- `clip_id`: stable run/segment identifier

### Important NPZ Encoding Notes

The current planner-side live consumer reads:

- `t0_us` as `int(npz["t0_us"][0])`
- `fixed_delta_seconds` as `float(npz["fixed_delta_seconds"][0])`
- `clip_id` as `str(npz["clip_id"][0])`
- `camera_order` as `npz["camera_order"].tolist()`

Expected HTTP headers from the sensor container:

- `X-Sample-Sequence`
- `X-T0-US`
- `X-Clip-ID`

So the planner container must assume:

- metadata fields are NumPy arrays, not bare scalars
- `relative_timestamps` and `absolute_timestamps` describe the **actual selected frames**

## Existing Repo Pieces To Reuse

Do not start from scratch. The repo already contains a working live consumer path.

### Existing live sample fetch path

- sample fetch and HTTP `503` handling: [jetson_live_infer_alpamayo15.py:180](/root/TensorRT-Edge-LLM-v060/jetson_live_infer_alpamayo15.py:180)

### Existing sample staging path

- PNG / NPY staging: [jetson_live_infer_alpamayo15.py:213](/root/TensorRT-Edge-LLM-v060/jetson_live_infer_alpamayo15.py:213)

### Existing request JSON generation

- request builder: [jetson_live_infer_alpamayo15.py:276](/root/TensorRT-Edge-LLM-v060/jetson_live_infer_alpamayo15.py:276)

### Existing persistent inference path

- persistent client wrapper: [jetson_live_infer_alpamayo15.py:352](/root/TensorRT-Edge-LLM-v060/jetson_live_infer_alpamayo15.py:352)
- `llm_inference --persistentServer` command protocol: [jetson_live_infer_alpamayo15.py:430](/root/TensorRT-Edge-LLM-v060/jetson_live_infer_alpamayo15.py:430)

### Current runtime constraint

The runtime is not fully in-memory yet:

- FM still requires `ego_history_xyz_npy` / `ego_history_rot_npy` file paths: [alpamayoPostVlmRuntime.cpp:217](/root/TensorRT-Edge-LLM-v060/cpp/runtime/alpamayoPostVlmRuntime.cpp:217)

So Phase 1 should continue to use:

- PNG image staging
- NPY ego-history staging
- request JSON

## Recommended Implementation Phases

Do this in order. Do **not** jump to the final direct-memory API immediately.

### Phase 1: HTTP /latest + current staging

Implement a planner-side service that:

- polls sensor `/latest`
- validates samples
- drops old samples if inference is busy
- writes PNG / NPY / request JSON
- submits requests to persistent `llm_inference`
- parses the output JSON

Advantages:

- fastest bring-up
- reuses existing repo logic
- easiest to debug against offline assets

### Phase 2: planner service hardening

After Phase 1 is stable:

- replace polling loop with explicit worker service structure
- add health endpoints and metrics
- add output/result handoff interface for control bridge
- keep the runtime boundary the same

### Phase 3: direct planner API

After the above is stable:

- remove request JSON boundary
- remove PNG image staging
- remove NPY ego-history staging
- pass sample buffers directly into runtime

This phase requires runtime changes and is **not** the first target.

## Planner Container Responsibilities

The planner container should own the following.

### 1. Sample Receiver

Required behavior:

- poll sensor container `/latest`
- parse NPZ payload
- read metadata headers
- treat `HTTP 503` as “no new valid sample yet”

### 2. Sample Validation

Before staging or inference, validate:

- all required keys exist
- shapes match the fixed contract
- dtypes are acceptable
- camera order / camera IDs match the agreed mapping
- `ego_history_xyz[..., -1, :]` is near zero
- last `ego_history_rot` is near identity

If validation fails:

- do not run inference
- mark the sample invalid in health output
- keep last valid planner result if policy allows

### 3. Latest-Only Scheduling

This is mandatory.

The planner container must **not** allow an unbounded backlog of stale samples.

Policy:

- keep at most one pending sample while inference is busy
- if a newer sample arrives, replace the older pending sample
- prefer dropping old samples over processing stale ones

### 4. Runtime File Staging

Phase 1 still needs:

- image PNG files
- `ego_history_xyz.npy`
- `ego_history_rot.npy`
- request JSON

Use per-run staging directories under:

- `input/images/live_runtime`
- `input/ego/live_runtime`
- `input/requests/live_runtime`

or a cleaner container-local equivalent if the service is rewritten.

### 5. Persistent Inference Service

Required behavior:

- start `llm_inference --persistentServer` once
- keep it alive
- submit one request at a time
- restart it cleanly if it dies

### 6. Output Parsing

After inference:

- parse output JSON
- extract at least:
  - `output_text`
  - `pred_xyz`
  - `pred_rot`
  - timing block if present
  - source sample metadata (`sequence`, `t0_us`, `clip_id`)

Normalize this into a stable internal result object.

### 7. Downstream Handoff

Phase 1 target:

- write output JSON and internal parsed result
- make the latest parsed result available to a control bridge or replay/visualization process

Phase 2 target:

- expose a direct result handoff API or shared-memory result slot

### 8. Health / Metrics

Expose:

- last received sample sequence
- last received `t0_us`
- current sample age
- last successful inference completion time
- current inference busy / idle state
- dropped sample count
- invalid sample count
- persistent runtime alive / dead state

## Recommended Service Split Inside Planner Container

Even inside one container, keep the responsibilities separated logically.

Recommended components:

1. `sample_receiver`
   - polls `/latest`
   - validates incoming samples
   - enforces latest-only policy

2. `planner_runtime_service`
   - stages files
   - manages persistent `llm_inference`
   - parses outputs

3. `planner_health_service`
   - exposes status / metrics

4. optional `result_bridge`
   - writes latest planner result for control bridge / viewer / logger

## Suggested Files To Add

This is the suggested structure for the next Codex working in the planner container.

```text
planner_live/
  README.md
  planner_live_service.py
  sample_contract.py
  sample_validator.py
  persistent_runtime.py
  result_parser.py
  health_server.py
  tests/
    test_sample_validation.py
    test_latest_only.py
    test_persistent_runtime_restart.py
    test_end_to_end_live_sample.py
```

If this is implemented by extending `jetson_live_infer_alpamayo15.py` first, that is acceptable for Phase 1.

## Implementation Tasks

### A. Build planner-side sample receiver

- configure sensor container base URL
- poll `/latest`
- handle `503` cleanly
- parse NPZ payload
- capture metadata headers

### B. Implement sample validator

- enforce required keys
- enforce shapes
- enforce dtype compatibility
- enforce camera ID / order mapping
- reject malformed samples before inference

### C. Add latest-only scheduling

- ensure stale samples do not accumulate
- keep only the newest pending sample
- count dropped / replaced samples

### D. Keep or refactor runtime staging

- write planner-facing PNG files
- write `ego_history_xyz.npy`
- write `ego_history_rot.npy`
- generate request JSON

### E. Wrap persistent inference

- start persistent `llm_inference`
- submit requests serially
- detect process death
- restart automatically if possible

### F. Parse outputs

- parse output JSON into a stable internal result structure
- include source metadata
- include timing
- expose latest valid result for downstream use

### G. Health endpoint

- publish service state
- publish last sample / last inference metadata
- publish counters for invalid / dropped samples

## Test Plan

The following tests should be implemented before calling the planner container "usable".

### 1. Sample Receiver Tests

- `/latest` success path works
- `503` path does not crash the loop
- repeated polls do not leak memory

### 2. Sample Validation Tests

- missing key is rejected
- wrong camera order is rejected
- wrong shape is rejected
- malformed metadata arrays are rejected

### 3. Latest-Only Policy Tests

- when inference is slow, stale samples are dropped
- the newest pending sample replaces the older pending sample
- drop counters are visible

### 4. Runtime Staging Tests

- PNG files are written correctly
- NPY files are written correctly
- request JSON matches the current runtime schema

### 5. Persistent Runtime Tests

- persistent server becomes ready
- one request runs successfully
- multiple requests run sequentially
- runtime restart path works if the process exits unexpectedly

### 6. End-to-End Live Tests

- planner container consumes a real `/latest` sample
- one full inference completes
- output JSON is written successfully
- parsed result object is produced successfully

### 7. Long-Run Stability Tests

- run for at least `30 min`
- no process wedging
- no uncontrolled backlog growth
- no unbounded temp file growth

## Acceptance Criteria

The Phase 1 planner container is accepted when:

1. It can consume Phase 1 `/latest` NPZ samples from the sensor container.
2. It validates malformed samples before inference.
3. It runs persistent Alpamayo inference successfully.
4. It uses a latest-only policy instead of building stale backlog.
5. It exposes useful health / freshness state.
6. It can run continuously for at least `30 min` without manual intervention.

## Risks / Things To Watch

### 1. Silent sample drift

If the sample contract changes without validation, planner results may degrade without obvious errors.

### 2. Backlog growth

This is likely the most important runtime risk in the current latency regime.

### 3. Staging file growth

If live run directories are not cleaned properly, disk usage will grow over time.

### 4. Persistent runtime death

The service must handle `llm_inference` failure cleanly and visibly.

### 5. Over-optimizing too early

Do not start by removing JSON / PNG / NPY boundaries. First make the end-to-end path stable.

## Recommended First Deliverables

The next Codex should aim to produce these in order:

1. planner live service entrypoint
2. sample validator
3. latest-only scheduler
4. persistent runtime wrapper
5. parsed planner result object
6. planner health endpoint
7. one successful live end-to-end run against a sensor `/latest` server

## Immediate Next Step

The best next move is:

1. create a planner-side service that reuses the current live sample fetch + staging + persistent inference path
2. add explicit sample validation in front of inference
3. add latest-only backlog protection
4. verify one full end-to-end live run against the sensor container

Once that works, the team can decide whether to harden the service structure further or move toward direct-memory integration.
