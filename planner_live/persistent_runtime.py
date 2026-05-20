from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

from .result_parser import PlannerResult, parse_planner_result
from .sample_contract import SampleMetadata


@dataclass(slots=True)
class PlannerRuntimeConfig:
    llm_inference_bin: Path
    plugin_lib: Path
    engine_dir: Path
    multimodal_engine_dir: Path
    fm_engine: Path
    staging_root: Path
    output_root: Path
    action_space_constants_json: Path | None = None
    nav_text: str | None = None
    nav_guidance_weight: float = 3.0
    traj_token_offset: int = 3000
    diffusion_seed: int = 42
    diffusion_num_steps: int = 2
    max_generate_length: int = 20
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 1
    warmup: int = 0
    keep_last_only: bool = True
    enable_dashboard: bool = False
    alpamayo_fm_use_prefill_kv: bool = False
    quiet_llm_logs: bool = False

    @property
    def image_dir(self) -> Path:
        return self.staging_root / "images"

    @property
    def ego_dir(self) -> Path:
        return self.staging_root / "ego"

    @property
    def request_dir(self) -> Path:
        return self.staging_root / "requests"

    @property
    def runs_dir(self) -> Path:
        return self.output_root / "runs"

    @property
    def results_dir(self) -> Path:
        return self.output_root / "results"

    @property
    def trajectories_dir(self) -> Path:
        return self.output_root / "trajectories"

    @property
    def dashboards_dir(self) -> Path:
        return self.output_root / "dashboards"


def _live_helpers() -> dict[str, Any]:
    from jetson_live_infer_alpamayo15 import (
        PersistentLLMInferenceClient,
        build_runtime_request,
        extract_trajectory_artifacts,
        load_action_space_constants,
        remove_tree_contents,
        save_live_dashboard,
        write_sample_files,
    )

    return {
        "PersistentLLMInferenceClient": PersistentLLMInferenceClient,
        "build_runtime_request": build_runtime_request,
        "extract_trajectory_artifacts": extract_trajectory_artifacts,
        "load_action_space_constants": load_action_space_constants,
        "remove_tree_contents": remove_tree_contents,
        "save_live_dashboard": save_live_dashboard,
        "write_sample_files": write_sample_files,
    }


@dataclass(slots=True)
class DashboardJob:
    sample: dict[str, Any]
    output_file: Path
    run_name: str
    metadata_dict: dict[str, Any]
    result_path: Path
    latest_result_path: Path


class AsyncDashboardWriter:
    def __init__(
        self,
        *,
        dashboard_dir: Path,
        save_live_dashboard: Callable[..., Path],
    ) -> None:
        self.dashboard_dir = dashboard_dir
        self.save_live_dashboard = save_live_dashboard
        self._cond = threading.Condition()
        self._pending: DashboardJob | None = None
        self._closed = False
        self._last_error: str | None = None
        self._last_completed_sequence: int | None = None
        self._thread = threading.Thread(target=self._run, name="planner-dashboard-writer", daemon=True)
        self._thread.start()

    def submit(self, job: DashboardJob) -> None:
        with self._cond:
            self._pending = job
            self._cond.notify()

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def snapshot(self) -> dict[str, Any]:
        with self._cond:
            pending_sequence = None
            if self._pending is not None:
                pending_sequence = int(self._pending.metadata_dict["sequence"])
            return {
                "dashboard_writer_alive": self._thread.is_alive(),
                "dashboard_pending_sequence": pending_sequence,
                "dashboard_last_completed_sequence": self._last_completed_sequence,
                "dashboard_last_error": self._last_error,
            }

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._closed:
                    self._cond.wait()
                if self._pending is None and self._closed:
                    return
                job = self._pending
                self._pending = None

            assert job is not None
            try:
                dashboard_path = self.save_live_dashboard(
                    sample=job.sample,
                    output_file=job.output_file,
                    profile_file=None,
                    dashboard_dir=self.dashboard_dir,
                    run_name=job.run_name,
                    metadata=job.metadata_dict,
                )
                self._patch_result_dashboard_path(job, dashboard_path)
                with self._cond:
                    self._last_error = None
                    self._last_completed_sequence = int(job.metadata_dict["sequence"])
            except Exception as exc:
                with self._cond:
                    self._last_error = str(exc)

    @staticmethod
    def _patch_result_dashboard_path(job: DashboardJob, dashboard_path: Path) -> None:
        for path in (job.result_path, job.latest_result_path):
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text())
            except Exception:
                continue
            if path == job.latest_result_path and int(payload.get("sequence", -1)) != int(job.metadata_dict["sequence"]):
                continue
            payload["dashboard_png_path"] = str(dashboard_path)
            path.write_text(json.dumps(payload, indent=2))


class PlannerRuntimeService:
    def __init__(self, config: PlannerRuntimeConfig) -> None:
        self.config = config
        self.helpers = _live_helpers()
        self._client = None
        self._action_space_constants = self.helpers["load_action_space_constants"](config.action_space_constants_json)
        self._dashboard_writer = (
            AsyncDashboardWriter(
                dashboard_dir=config.dashboards_dir,
                save_live_dashboard=self.helpers["save_live_dashboard"],
            )
            if config.enable_dashboard
            else None
        )
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for path in (
            self.config.image_dir,
            self.config.ego_dir,
            self.config.request_dir,
            self.config.runs_dir,
            self.config.results_dir,
            self.config.trajectories_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
        if self.config.enable_dashboard:
            self.config.dashboards_dir.mkdir(parents=True, exist_ok=True)

    def _build_client_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            plugin_lib=self.config.plugin_lib,
            llm_inference_bin=self.config.llm_inference_bin,
            engine_dir=self.config.engine_dir,
            multimodal_engine_dir=self.config.multimodal_engine_dir,
            fm_engine=self.config.fm_engine,
            warmup=self.config.warmup,
            nav_text=self.config.nav_text,
            alpamayo_fm_use_prefill_kv=self.config.alpamayo_fm_use_prefill_kv,
            quiet_llm_logs=self.config.quiet_llm_logs,
            dump_profile=False,
        )

    def _restart_client(self) -> None:
        self._close_client()
        self._client = self.helpers["PersistentLLMInferenceClient"](self._build_client_args())
        self._client.start()

    def start(self) -> None:
        if self._client is None:
            self._client = self.helpers["PersistentLLMInferenceClient"](self._build_client_args())
        self._client.start()

    def _close_client(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def close(self) -> None:
        self._close_client()
        if self._dashboard_writer is not None:
            self._dashboard_writer.close()
            self._dashboard_writer = None

    def dashboard_snapshot(self) -> dict[str, Any]:
        if self._dashboard_writer is None:
            return {"dashboard_writer_enabled": False}
        snapshot = self._dashboard_writer.snapshot()
        snapshot["dashboard_writer_enabled"] = True
        return snapshot

    def is_runtime_alive(self) -> bool:
        if self._client is None or self._client.process is None:
            return False
        return self._client.process.poll() is None

    def _run_persistent_request(self, request_path: Path, output_file: Path) -> None:
        for attempt in range(2):
            try:
                self.start()
                assert self._client is not None
                self._client.run(request_path=request_path, output_file=output_file)
                return
            except Exception:
                if attempt == 1:
                    raise
                self._restart_client()

    def process_sample(self, sample: dict[str, Any], metadata: SampleMetadata) -> PlannerResult:
        self._ensure_dirs()
        if self.config.keep_last_only:
            self.helpers["remove_tree_contents"](self.config.image_dir)
            self.helpers["remove_tree_contents"](self.config.ego_dir)
            self.helpers["remove_tree_contents"](self.config.request_dir)

        run_name = f"seq{metadata.sequence:06d}_t0_{metadata.t0_us}"

        _, xyz_path, rot_path = self.helpers["write_sample_files"](sample, self.config.image_dir, self.config.ego_dir)
        request_path = self.config.request_dir / f"request_{run_name}.json"
        latest_request_path = self.config.request_dir / "latest_request.json"
        self.helpers["build_runtime_request"](
            sample=sample,
            xyz_path=xyz_path,
            rot_path=rot_path,
            image_dir=self.config.image_dir,
            output_request=request_path,
            action_space_constants=self._action_space_constants,
            nav_text=self.config.nav_text,
            nav_guidance_weight=self.config.nav_guidance_weight,
            traj_token_offset=self.config.traj_token_offset,
            diffusion_seed=self.config.diffusion_seed,
            diffusion_num_steps=self.config.diffusion_num_steps,
            max_generate_length=self.config.max_generate_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
        )
        latest_request_path.write_text(request_path.read_text())

        output_file = self.config.runs_dir / f"output_{run_name}.json"
        self._run_persistent_request(request_path=request_path, output_file=output_file)

        latest_output_path = self.config.runs_dir / "latest_output.json"
        latest_output_path.write_text(output_file.read_text())

        metadata_dict = {
            "sequence": metadata.sequence,
            "t0_us": metadata.t0_us,
            "clip_id": metadata.clip_id,
        }
        traj_json, traj_xyz_npy, traj_rot_npy = self.helpers["extract_trajectory_artifacts"](
            output_file=output_file,
            trajectory_dir=self.config.trajectories_dir,
            run_name=run_name,
            metadata=metadata_dict,
            sample=sample,
        )

        result = parse_planner_result(
            output_file=output_file,
            metadata=metadata,
            trajectory_json_path=traj_json,
            trajectory_xyz_npy_path=traj_xyz_npy,
            trajectory_rot_npy_path=traj_rot_npy,
            dashboard_png_path=None,
        )

        result_path = self.config.results_dir / f"result_{run_name}.json"
        latest_result_path = self.config.results_dir / "latest_result.json"
        payload = json.dumps(result.to_dict(), indent=2)
        result_path.write_text(payload)
        latest_result_path.write_text(payload)
        if self._dashboard_writer is not None:
            self._dashboard_writer.submit(
                DashboardJob(
                    sample=sample,
                    output_file=output_file,
                    run_name=run_name,
                    metadata_dict=metadata_dict,
                    result_path=result_path,
                    latest_result_path=latest_result_path,
                )
            )
        return result
