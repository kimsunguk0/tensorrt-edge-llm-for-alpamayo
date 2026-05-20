# Planner Live Service

Phase 1 planner-container service for consuming `/latest` sensor samples,
validating them, staging the current runtime files, running persistent
`llm_inference`, and publishing a stable latest planner result.

## Entrypoint

```bash
python -m planner_live.planner_live_service \
  --server-url http://127.0.0.1:8765 \
  --engine-dir /workspace/models/alpamayo_runtime/engines/alpa1.5 \
  --multimodal-engine-dir /workspace/models/alpamayo_runtime/engines/alpa1.5_visual_fp8_rebuild \
  --fm-engine /workspace/models/alpamayo_runtime/fm/flowmatching_20260424_trt_fp8/engine_thor/teacher_structured_student_reflow_consistency_step4_mlp6144_one_step_fp8_s3328_thor.plan \
  --diffusion-num-steps 2 \
  --alpamayo-fm-use-prefill-kv \
  --enable-udp-bridge \
  --udp-host 127.0.0.1 \
  --udp-port 5001
```

## What It Does

- polls `/latest`
- validates the sample contract before inference
- keeps only the newest pending sample while inference is busy
- stages PNG / NPY / request JSON files
- keeps a persistent `llm_inference` process alive
- writes output JSON, trajectory artifacts, and a stable latest result JSON
- can optionally bridge the latest planner result to the control-team UDP packet format
- serves health and freshness JSON on `/healthz` and `/status`
- serves an auto-refresh browser viewer on `/viewer`

## Output Layout

By default, service outputs go under:

```text
output/planner_live/
  runs/
  results/
  trajectories/
  dashboards/
  staging/
```

## Live Viewer

When the service is running, open:

```text
http://localhost:8780/viewer
```

The page auto-refreshes the latest dashboard PNG and parsed result JSON every second.
