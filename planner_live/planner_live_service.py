from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import threading
import time
from pathlib import Path
from typing import Any, Generic, TypeVar

from fm_model_defaults import preferred_fm_engine_candidates

from .gnss_serial import GnssSerialConfig, GnssSerialReader
from .health_server import HealthServer
from .persistent_runtime import PlannerRuntimeConfig, PlannerRuntimeService
from .result_bridge import UdpBridgeConfig, UdpResultBridge
from .result_parser import PlannerResult
from .sample_contract import SampleEnvelope, SampleMetadata
from .sample_validator import ValidationError, validate_live_sample


T = TypeVar("T")


def _first_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class LatestOnlyBuffer(Generic[T]):
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._pending: T | None = None
        self._closed = False

    def offer(self, item: T) -> T | None:
        with self._cond:
            replaced = self._pending
            self._pending = item
            self._cond.notify()
            return replaced

    def take(self, timeout: float | None = None) -> T | None:
        with self._cond:
            if self._pending is None and not self._closed:
                self._cond.wait(timeout=timeout)
            if self._pending is None:
                return None
            item = self._pending
            self._pending = None
            return item

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def discard_if(self, predicate: Callable[[T], bool]) -> T | None:
        with self._cond:
            if self._pending is None or not predicate(self._pending):
                return None
            item = self._pending
            self._pending = None
            return item

    def peek(self) -> T | None:
        with self._cond:
            return self._pending


@dataclass(slots=True)
class PlannerServiceConfig:
    server_url: str
    sample_endpoint: str
    timeout: float
    poll_interval: float
    health_host: str
    health_port: int
    runtime: PlannerRuntimeConfig
    udp_bridge: UdpBridgeConfig | None = None
    gnss_serial: GnssSerialConfig | None = None
    gnss_required: bool = False
    gnss_history_len: int = 16
    gnss_dt_s: float = 0.1
    manual_trigger_only: bool = False


class PlannerServiceState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.service_running = False
        self.receiver_thread_alive = False
        self.worker_thread_alive = False
        self.runtime_alive = False
        self.inference_busy = False
        self.last_received_sequence: int | None = None
        self.last_received_t0_us: int | None = None
        self.last_received_clip_id: str | None = None
        self.last_received_at_unix: float | None = None
        self.last_success_sequence: int | None = None
        self.last_success_t0_us: int | None = None
        self.last_success_clip_id: str | None = None
        self.last_success_completed_at_unix: float | None = None
        self.current_pending_sequence: int | None = None
        self.dropped_sample_count = 0
        self.pre_inference_refresh_count = 0
        self.pre_inference_refresh_from_sequence: int | None = None
        self.pre_inference_refresh_to_sequence: int | None = None
        self.pre_inference_refresh_last_error: str | None = None
        self.invalid_sample_count = 0
        self.total_inference_count = 0
        self.last_error: str | None = None
        self.latest_result_path: str | None = None
        self.manual_trigger_only = False
        self.manual_run_busy = False
        self.manual_run_count = 0
        self.manual_last_status: str | None = None
        self.manual_last_error: str | None = None
        self.manual_last_completed_at_unix: float | None = None

    def _update(self, updater: Callable[["PlannerServiceState"], None]) -> None:
        with self._lock:
            updater(self)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.time()
            received_age_ms = None
            if self.last_received_at_unix is not None:
                received_age_ms = max((now - self.last_received_at_unix) * 1000.0, 0.0)
            last_success_age_ms = None
            if self.last_success_completed_at_unix is not None:
                last_success_age_ms = max((now - self.last_success_completed_at_unix) * 1000.0, 0.0)
            return {
                "service_running": self.service_running,
                "receiver_thread_alive": self.receiver_thread_alive,
                "worker_thread_alive": self.worker_thread_alive,
                "runtime_alive": self.runtime_alive,
                "inference_busy": self.inference_busy,
                "last_received_sequence": self.last_received_sequence,
                "last_received_t0_us": self.last_received_t0_us,
                "last_received_clip_id": self.last_received_clip_id,
                "last_received_age_ms": received_age_ms,
                "last_success_sequence": self.last_success_sequence,
                "last_success_t0_us": self.last_success_t0_us,
                "last_success_clip_id": self.last_success_clip_id,
                "last_success_age_ms": last_success_age_ms,
                "current_pending_sequence": self.current_pending_sequence,
                "dropped_sample_count": self.dropped_sample_count,
                "pre_inference_refresh_count": self.pre_inference_refresh_count,
                "pre_inference_refresh_from_sequence": self.pre_inference_refresh_from_sequence,
                "pre_inference_refresh_to_sequence": self.pre_inference_refresh_to_sequence,
                "pre_inference_refresh_last_error": self.pre_inference_refresh_last_error,
                "invalid_sample_count": self.invalid_sample_count,
                "total_inference_count": self.total_inference_count,
                "last_error": self.last_error,
                "latest_result_path": self.latest_result_path,
                "manual_trigger_only": self.manual_trigger_only,
                "manual_run_busy": self.manual_run_busy,
                "manual_run_count": self.manual_run_count,
                "manual_last_status": self.manual_last_status,
                "manual_last_error": self.manual_last_error,
                "manual_last_completed_at_unix": self.manual_last_completed_at_unix,
            }


class PlannerLiveService:
    def __init__(self, config: PlannerServiceConfig) -> None:
        self.config = config
        self.runtime = PlannerRuntimeService(config.runtime)
        self.state = PlannerServiceState()
        self.result_bridge = UdpResultBridge(config.udp_bridge) if config.udp_bridge is not None else None
        self.gnss_reader = GnssSerialReader(config.gnss_serial) if config.gnss_serial is not None else None
        self.pending = LatestOnlyBuffer[SampleEnvelope]()
        self.stop_event = threading.Event()
        self.receiver_thread = threading.Thread(target=self._receiver_loop, name="planner-sample-receiver", daemon=True)
        self.worker_thread = threading.Thread(target=self._worker_loop, name="planner-runtime-worker", daemon=True)
        self._inference_lock = threading.Lock()
        self.health_server = HealthServer(
            config.health_host,
            config.health_port,
            self._snapshot,
            self._artifact_paths,
            self.manual_run_once,
        )
        self._last_seen_sequence: int | None = None
        self._last_seen_lock = threading.Lock()
        self.state._update(lambda s: setattr(s, "manual_trigger_only", config.manual_trigger_only))

    def start(self) -> None:
        self.state._update(lambda s: setattr(s, "service_running", True))
        if self.gnss_reader is not None:
            self.gnss_reader.start()
        if self.result_bridge is not None and not self.config.manual_trigger_only:
            self.result_bridge.start()
        self.health_server.start()
        if not self.config.manual_trigger_only:
            self.receiver_thread.start()
            self.worker_thread.start()

    def stop(self) -> None:
        if self.stop_event.is_set() and not self.state.snapshot().get("service_running", False):
            return
        self.stop_event.set()
        self.pending.close()
        if self.receiver_thread.is_alive():
            self.receiver_thread.join(timeout=5.0)
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        if self.result_bridge is not None:
            self.result_bridge.stop()
        if self.gnss_reader is not None:
            self.gnss_reader.stop()
        self.runtime.close()
        self.health_server.stop()
        self.state._update(lambda s: setattr(s, "service_running", False))

    def wait_forever(self) -> None:
        try:
            while not self.stop_event.is_set():
                time.sleep(0.5)
        finally:
            self.stop()

    def _receiver_loop(self) -> None:
        self.state._update(lambda s: setattr(s, "receiver_thread_alive", True))
        try:
            while not self.stop_event.is_set():
                try:
                    envelope = self._fetch_and_validate_sample(allow_duplicate=False)
                except ValidationError:
                    time.sleep(self.config.poll_interval)
                    continue
                except Exception as exc:
                    self.state._update(lambda s: setattr(s, "last_error", f"sample fetch failed: {exc}"))
                    time.sleep(self.config.poll_interval)
                    continue

                if envelope is None:
                    time.sleep(self.config.poll_interval)
                    continue

                replaced = self.pending.offer(envelope)
                self.state._update(lambda s: setattr(s, "current_pending_sequence", envelope.metadata.sequence))
                if replaced is not None:
                    self.state._update(
                        lambda s: setattr(s, "dropped_sample_count", s.dropped_sample_count + 1)
                    )
                time.sleep(self.config.poll_interval)
        finally:
            self.state._update(lambda s: setattr(s, "receiver_thread_alive", False))

    def _worker_loop(self) -> None:
        self.state._update(lambda s: setattr(s, "worker_thread_alive", True))
        try:
            while not self.stop_event.is_set():
                envelope = self.pending.take(timeout=0.5)
                if envelope is None:
                    self.state._update(lambda s: setattr(s, "runtime_alive", self.runtime.is_runtime_alive()))
                    continue

                self.state._update(
                    lambda s: (
                        setattr(s, "current_pending_sequence", None),
                        setattr(s, "runtime_alive", self.runtime.is_runtime_alive()),
                    )
                )

                envelope = self._refresh_sample_before_inference(envelope)
                try:
                    result = self._run_inference(envelope)
                except Exception as exc:
                    self.state._update(
                        lambda s: (
                            setattr(s, "last_error", f"runtime failure seq={envelope.metadata.sequence}: {exc}"),
                            setattr(s, "runtime_alive", self.runtime.is_runtime_alive()),
                        )
                    )
                    continue

                self._record_success(result)
        finally:
            self.state._update(lambda s: setattr(s, "worker_thread_alive", False))

    def _fetch_and_validate_sample(self, *, allow_duplicate: bool) -> SampleEnvelope | None:
        from jetson_live_infer_alpamayo15 import fetch_latest_sample

        sample, metadata_obj = fetch_latest_sample(
            self.config.server_url,
            self.config.sample_endpoint,
            self.config.timeout,
        )
        if sample is None:
            return None

        metadata = SampleMetadata(
            sequence=int(metadata_obj["sequence"]),
            t0_us=int(metadata_obj["t0_us"]),
            clip_id=str(metadata_obj["clip_id"]),
            received_at_unix=time.time(),
            source_headers={
                "X-Sample-Sequence": str(metadata_obj["sequence"]),
                "X-T0-US": str(metadata_obj["t0_us"]),
                "X-Clip-ID": str(metadata_obj["clip_id"]),
            },
        )

        if not allow_duplicate:
            with self._last_seen_lock:
                if self._last_seen_sequence is not None and metadata.sequence <= self._last_seen_sequence:
                    return None
                self._last_seen_sequence = metadata.sequence

        sample = self._apply_gnss_history(sample, metadata)

        self.state._update(
            lambda s: (
                setattr(s, "last_received_sequence", metadata.sequence),
                setattr(s, "last_received_t0_us", metadata.t0_us),
                setattr(s, "last_received_clip_id", metadata.clip_id),
                setattr(s, "last_received_at_unix", metadata.received_at_unix),
            )
        )

        try:
            validated_sample = validate_live_sample(sample)
        except ValidationError as exc:
            self.state._update(
                lambda s: (
                    setattr(s, "invalid_sample_count", s.invalid_sample_count + 1),
                    setattr(s, "last_error", f"invalid sample seq={metadata.sequence}: {exc}"),
                )
            )
            raise

        return SampleEnvelope(sample=validated_sample, metadata=metadata)

    def _mark_sequence_seen(self, sequence: int) -> None:
        with self._last_seen_lock:
            if self._last_seen_sequence is None or sequence > self._last_seen_sequence:
                self._last_seen_sequence = sequence

    def _refresh_sample_before_inference(self, envelope: SampleEnvelope) -> SampleEnvelope:
        try:
            refreshed = self._fetch_and_validate_sample(allow_duplicate=True)
        except Exception as exc:
            self.state._update(
                lambda s: setattr(s, "pre_inference_refresh_last_error", f"seq={envelope.metadata.sequence}: {exc}")
            )
            return envelope

        if refreshed is None or refreshed.metadata.sequence <= envelope.metadata.sequence:
            return envelope

        self._mark_sequence_seen(refreshed.metadata.sequence)
        discarded = self.pending.discard_if(
            lambda pending: pending.metadata.sequence <= refreshed.metadata.sequence
        )
        pending_now = self.pending.peek()
        self.state._update(
            lambda s: (
                setattr(s, "pre_inference_refresh_count", s.pre_inference_refresh_count + 1),
                setattr(s, "pre_inference_refresh_from_sequence", envelope.metadata.sequence),
                setattr(s, "pre_inference_refresh_to_sequence", refreshed.metadata.sequence),
                setattr(s, "pre_inference_refresh_last_error", None),
                setattr(
                    s,
                    "current_pending_sequence",
                    pending_now.metadata.sequence if pending_now is not None else None,
                ),
                setattr(s, "dropped_sample_count", s.dropped_sample_count + (1 if discarded is not None else 0)),
            )
        )
        return refreshed

    def _apply_gnss_history(self, sample: dict[str, Any], metadata: SampleMetadata) -> dict[str, Any]:
        if self.gnss_reader is None:
            return sample

        history = self.gnss_reader.build_ego_history(
            t0_utc_ns=int(metadata.t0_us) * 1000,
            history_len=self.config.gnss_history_len,
            dt_s=self.config.gnss_dt_s,
        )
        if history is None:
            message = "GNSS ego history unavailable: waiting for enough valid fixes"
            self.state._update(lambda s: setattr(s, "last_error", message))
            if self.config.gnss_required:
                raise ValidationError([message])
            return sample

        ego_history_xyz, ego_history_rot, info = history
        patched = dict(sample)
        patched["ego_history_xyz"] = ego_history_xyz
        patched["ego_history_rot"] = ego_history_rot
        patched["gnss_history_info"] = info
        return patched

    def _run_inference(self, envelope: SampleEnvelope) -> PlannerResult:
        with self._inference_lock:
            self.state._update(
                lambda s: (
                    setattr(s, "inference_busy", True),
                    setattr(s, "runtime_alive", self.runtime.is_runtime_alive()),
                )
            )
            try:
                return self.runtime.process_sample(envelope.sample, envelope.metadata)
            finally:
                self.state._update(
                    lambda s: (
                        setattr(s, "inference_busy", False),
                        setattr(s, "runtime_alive", self.runtime.is_runtime_alive()),
                    )
                )

    def manual_run_once(self) -> dict[str, Any]:
        self.state._update(
            lambda s: (
                setattr(s, "manual_run_busy", True),
                setattr(s, "manual_last_status", "running"),
                setattr(s, "manual_last_error", None),
            )
        )
        try:
            envelope = self._fetch_and_validate_sample(allow_duplicate=True)
            if envelope is None:
                raise RuntimeError("latest sample unavailable (server returned 503 no valid sample yet)")

            result = self._run_inference(envelope)

            udp_info: dict[str, Any] | None = None
            if self.result_bridge is not None and self.result_bridge.config.enabled:
                udp_info = self.result_bridge.send_result_once(result)

            self._record_success(result, publish_bridge=False)
            self.state._update(
                lambda s: (
                    setattr(s, "manual_run_count", s.manual_run_count + 1),
                    setattr(s, "manual_last_status", "completed"),
                    setattr(s, "manual_last_completed_at_unix", time.time()),
                    setattr(s, "manual_last_error", None),
                )
            )
            return {
                "ok": True,
                "sequence": result.sequence,
                "t0_us": result.t0_us,
                "clip_id": result.clip_id,
                "udp": udp_info,
                "latest_result_path": str(self.runtime.config.results_dir / "latest_result.json"),
            }
        except Exception as exc:
            self.state._update(
                lambda s: (
                    setattr(s, "manual_last_status", "failed"),
                    setattr(s, "manual_last_error", str(exc)),
                    setattr(s, "last_error", f"manual run failed: {exc}"),
                )
            )
            return {"ok": False, "error": str(exc)}
        finally:
            self.state._update(lambda s: setattr(s, "manual_run_busy", False))

    def _record_success(self, result: PlannerResult, *, publish_bridge: bool = True) -> None:
        if publish_bridge and self.result_bridge is not None:
            self.result_bridge.publish_result(result)
        self.state._update(
            lambda s: (
                setattr(s, "runtime_alive", self.runtime.is_runtime_alive()),
                setattr(s, "last_success_sequence", result.sequence),
                setattr(s, "last_success_t0_us", result.t0_us),
                setattr(s, "last_success_clip_id", result.clip_id),
                setattr(s, "last_success_completed_at_unix", result.completed_at_unix),
                setattr(s, "latest_result_path", str(self.runtime.config.results_dir / "latest_result.json")),
                setattr(s, "total_inference_count", s.total_inference_count + 1),
                setattr(s, "last_error", None),
            )
        )

    def _snapshot(self) -> dict[str, Any]:
        snapshot = self.state.snapshot()
        snapshot.update(self.runtime.dashboard_snapshot())
        if self.result_bridge is not None:
            snapshot.update(self.result_bridge.snapshot())
        if self.gnss_reader is not None:
            snapshot["gnss"] = self.gnss_reader.snapshot()
        else:
            snapshot["gnss"] = {"enabled": False}
        return snapshot

    def _artifact_paths(self) -> dict[str, Path | None]:
        latest_dashboard = None
        if self.runtime.config.enable_dashboard:
            latest_dashboard = self.runtime.config.dashboards_dir / "latest_dashboard.png"
        return {
            "latest_dashboard": latest_dashboard,
            "latest_result": self.runtime.config.results_dir / "latest_result.json",
            "latest_trajectory": self.runtime.config.trajectories_dir / "latest_trajectory.json",
        }


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parent.parent
    default_models_root = Path("/workspace/models/alpamayo_runtime")
    default_output_root = repo_root / "output" / "planner_live"

    parser = argparse.ArgumentParser(
        description="Planner-container live service with latest-only scheduling and persistent llm_inference."
    )
    parser.add_argument("--server-url", default="http://127.0.0.1:8765")
    parser.add_argument("--sample-endpoint", default="/latest")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--health-host", default="0.0.0.0")
    parser.add_argument("--health-port", type=int, default=8780)
    parser.add_argument(
        "--llm-inference-bin",
        type=Path,
        default=repo_root / "build" / "examples" / "llm" / "llm_inference",
    )
    parser.add_argument(
        "--plugin-lib",
        type=Path,
        default=repo_root / "build" / "libNvInfer_edgellm_plugin.so",
    )
    parser.add_argument(
        "--engine-dir",
        type=Path,
        default=_first_existing_path(
            [
                default_models_root / "engines" / "alpa1.5",
                Path("/alpamayo_vlm_engines/alpa1.5"),
            ]
        ),
    )
    parser.add_argument(
        "--multimodal-engine-dir",
        type=Path,
        default=_first_existing_path(
            [
                default_models_root / "engines" / "alpa1.5_visual_fp8_rebuild",
                default_models_root / "engines" / "alpa1.5",
                Path("/alpamayo_vlm_engines/alpa1.5"),
            ]
        ),
    )
    parser.add_argument(
        "--fm-engine",
        type=Path,
        default=_first_existing_path(
            preferred_fm_engine_candidates(default_models_root)
            + [
                Path("/root/test/output/alpamayo15_fm_one_step_fp16_true/alpamayo15_fm_one_step_fp16_true_thor.plan"),
                Path("/root/test/output/alpamayo15_fm_one_step_mxfp8/alpamayo15_fm_one_step_mxfp8_thor.plan"),
            ]
        ),
    )
    parser.add_argument("--output-root", type=Path, default=default_output_root)
    parser.add_argument("--action-space-constants-json", type=Path, default=None)
    parser.add_argument("--nav-text", default=None)
    parser.add_argument("--nav-guidance-weight", type=float, default=3.0)
    parser.add_argument("--traj-token-offset", type=int, default=3000)
    parser.add_argument("--diffusion-seed", type=int, default=42)
    parser.add_argument("--diffusion-num-steps", type=int, default=2)
    parser.add_argument("--max-generate-length", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--alpamayo-fm-use-prefill-kv",
        action="store_true",
        help="Feed backbone prefill KV directly to FM and skip CoT decode before FM.",
    )
    parser.add_argument("--enable-dashboard", action="store_true")
    parser.add_argument(
        "--quiet-llm-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hide noisy non-JSON llm_inference stdout lines while keeping planner_live logs.",
    )
    parser.add_argument("--enable-udp-bridge", action="store_true")
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5001)
    parser.add_argument(
        "--udp-payload-mode",
        choices=["binary_alpa", "text_json"],
        default="binary_alpa",
        help="binary_alpa sends compact ALPA packets; text_json matches the older viewer JSON UDP flow.",
    )
    parser.add_argument(
        "--udp-send-mode",
        choices=["periodic", "on_result"],
        default="periodic",
        help="periodic repeats the latest plan at udp-rate-hz; on_result sends once when each inference completes.",
    )
    parser.add_argument("--udp-rate-hz", type=float, default=50.0)
    parser.add_argument("--udp-period-s", type=float, default=None)
    parser.add_argument("--udp-control-dt", type=float, default=0.02)
    parser.add_argument("--udp-control-points", type=int, default=16)
    parser.add_argument("--udp-full-plan", action="store_true")
    parser.add_argument(
        "--udp-action-log-interval-s",
        type=float,
        default=3.0,
        help="Throttle interval for UDP accel/curvature preview logs. Use negative value to disable.",
    )
    parser.add_argument("--udp-action-log-points", type=int, default=16)
    parser.add_argument("--enable-gnss-serial", action="store_true")
    parser.add_argument("--gnss-device", default="/dev/ttyUSB0")
    parser.add_argument("--gnss-baud", type=int, default=115200)
    parser.add_argument(
        "--gnss-required",
        action="store_true",
        help="Skip inference until GNSS has enough valid fixes to build ego history.",
    )
    parser.add_argument("--gnss-history-len", type=int, default=16)
    parser.add_argument("--gnss-dt-s", type=float, default=0.1)
    parser.add_argument("--manual-trigger-only", action="store_true")
    return parser


def build_service_config(args: argparse.Namespace) -> PlannerServiceConfig:
    udp_rate_hz = float(args.udp_rate_hz)
    if args.udp_period_s is not None:
        if float(args.udp_period_s) <= 0.0:
            raise ValueError("--udp-period-s must be > 0")
        udp_rate_hz = 1.0 / float(args.udp_period_s)

    runtime_config = PlannerRuntimeConfig(
        llm_inference_bin=args.llm_inference_bin,
        plugin_lib=args.plugin_lib,
        engine_dir=args.engine_dir,
        multimodal_engine_dir=args.multimodal_engine_dir,
        fm_engine=args.fm_engine,
        staging_root=args.output_root / "staging",
        output_root=args.output_root,
        action_space_constants_json=args.action_space_constants_json,
        nav_text=args.nav_text,
        nav_guidance_weight=args.nav_guidance_weight,
        traj_token_offset=args.traj_token_offset,
        diffusion_seed=args.diffusion_seed,
        diffusion_num_steps=args.diffusion_num_steps,
        max_generate_length=args.max_generate_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        warmup=args.warmup,
        alpamayo_fm_use_prefill_kv=bool(args.alpamayo_fm_use_prefill_kv),
        enable_dashboard=args.enable_dashboard,
        quiet_llm_logs=bool(args.quiet_llm_logs),
    )
    return PlannerServiceConfig(
        server_url=args.server_url,
        sample_endpoint=args.sample_endpoint,
        timeout=args.timeout,
        poll_interval=args.poll_interval,
        health_host=args.health_host,
        health_port=args.health_port,
        runtime=runtime_config,
        udp_bridge=UdpBridgeConfig(
            enabled=bool(args.enable_udp_bridge),
            host=args.udp_host,
            port=args.udp_port,
            payload_mode=args.udp_payload_mode,
            send_mode=args.udp_send_mode,
            rate_hz=udp_rate_hz,
            control_dt_s=args.udp_control_dt,
            control_points=args.udp_control_points,
            full_plan=bool(args.udp_full_plan),
            action_log_interval_s=float(args.udp_action_log_interval_s),
            action_log_points=int(args.udp_action_log_points),
        ),
        gnss_serial=GnssSerialConfig(device=str(args.gnss_device), baud=int(args.gnss_baud))
        if bool(args.enable_gnss_serial)
        else None,
        gnss_required=bool(args.gnss_required),
        gnss_history_len=int(args.gnss_history_len),
        gnss_dt_s=float(args.gnss_dt_s),
        manual_trigger_only=bool(args.manual_trigger_only),
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    service = PlannerLiveService(build_service_config(args))
    service.start()
    try:
        service.wait_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
