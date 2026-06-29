from __future__ import annotations

import json
import socket
import time
import unittest

from planner_live.live_path_manager import (
    HealthzPoseSource,
    LivePathManager,
    LivePathManagerConfig,
    LivePose,
    _find_speed_mps,
    _find_utm,
    _find_yaw_rad,
)
from planner_live.result_bridge import UdpBridgeConfig, UdpResultBridge
from planner_live.result_parser import PlannerResult


def _identity_rot() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def make_result(sequence: int = 11, lateral_y_m: float = 0.0) -> PlannerResult:
    return PlannerResult(
        sequence=sequence,
        t0_us=time.time_ns() // 1000 - 200_000,
        clip_id="clip",
        completed_at_unix=time.time(),
        output_text="keep lane",
        fm_mode="single_branch",
        fm_status="completed",
        plan_dt_s=0.1,
        pred_xyz=[[float(idx) * 0.5, float(lateral_y_m), 0.0] for idx in range(1, 25)],
        pred_rot=[_identity_rot() for _ in range(24)],
        post_vlm_timing={},
        fm_timing={},
        output_json_path="/tmp/output.json",
    )


class FakePoseSource:
    label = "fake-pose"

    def __init__(self) -> None:
        self.pose = LivePose(
            x_m=100.0,
            y_m=200.0,
            yaw_rad=0.0,
            speed_mps=2.0,
            timestamp_utc_ns=time.time_ns(),
            receive_time_utc_ns=time.time_ns(),
            source="fake-pose",
        )

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def read_pose(self, timeout_s: float) -> LivePose:
        return self.pose


def make_pose(
    *,
    x_m: float = 100.0,
    y_m: float = 200.0,
    yaw_rad: float = 0.0,
    speed_mps: float = 2.0,
    receive_time_utc_ns: int | None = None,
) -> LivePose:
    now_ns = time.time_ns() if receive_time_utc_ns is None else receive_time_utc_ns
    return LivePose(
        x_m=x_m,
        y_m=y_m,
        yaw_rad=yaw_rad,
        speed_mps=speed_mps,
        timestamp_utc_ns=now_ns,
        receive_time_utc_ns=now_ns,
        source="fake-pose",
    )


class LivePathManagerTests(unittest.TestCase):
    def test_healthz_payload_parsing_finds_nested_pose_fields(self) -> None:
        payload = {
            "service": "sensor",
            "gnss": {"latest_fix": {"gnss_utm": [334007.1, 4147976.2, 42.0]}},
            "imu": {"euler_deg": [0.0, 0.0, 90.0]},
        }
        self.assertEqual(_find_utm(payload), [334007.1, 4147976.2, 42.0])
        yaw = _find_yaw_rad(payload)
        self.assertIsNotNone(yaw)
        self.assertAlmostEqual(float(yaw), 1.57079632679, places=5)

    def test_healthz_payload_parsing_finds_pose_inside_text_fields(self) -> None:
        payload = {
            "gnss": (
                "gnss_frames=5208 gnss_msg_hz=32.00 fix_type=1 "
                "vel=[-0.001640,0.001982,0.000000] "
                "utm=[333996.789273,4147977.649110,42.846500] utm_zone=52N"
            ),
            "imu": (
                "imu_count=4231 imu_sample_hz=26.00 "
                "euler_deg=[0.0000,0.0000,129.6000]"
            ),
        }
        self.assertEqual(_find_utm(payload), [333996.789273, 4147977.649110, 42.846500])
        yaw = _find_yaw_rad(payload)
        self.assertIsNotNone(yaw)
        self.assertAlmostEqual(float(yaw), 2.26194671058, places=5)
        speed = _find_speed_mps(payload)
        self.assertIsNotNone(speed)
        self.assertAlmostEqual(float(speed), 0.002572, places=5)

    def test_path_manager_blends_new_plan_from_previous_path(self) -> None:
        bridge = UdpResultBridge(UdpBridgeConfig(enabled=False, action_log_interval_s=-1.0))
        pose_source = FakePoseSource()
        manager = LivePathManager(
            LivePathManagerConfig(
                enabled=True,
                rate_hz=20.0,
                max_plan_age_s=5.0,
                min_plan_arc_m=1.0,
                min_remaining_distance_m=1.0,
                max_projection_distance_m=5.0,
                output_points=8,
                plan_blend_s=1.0,
                log_interval_s=-1.0,
            ),
            udp_bridge=bridge,
            pose_source=pose_source,
        )
        try:
            first = manager.publish_result(make_result(sequence=11, lateral_y_m=0.0))
            second = manager.publish_result(make_result(sequence=12, lateral_y_m=2.0))
            self.assertTrue(first["path_manager_plan_accepted"])
            self.assertTrue(second["path_manager_plan_accepted"])
            self.assertEqual(second["plan_blend_from_seq"], 11)
            self.assertGreater(second["plan_blend_s"], 0.0)
            self.assertIsNotNone(manager._plan)
            packet, debug = manager._build_packet(manager._plan, pose_source.pose)
        finally:
            bridge.stop()

        self.assertIsNotNone(packet)
        assert packet is not None
        self.assertTrue(packet["header"]["path_manager_plan_blend_active"])
        self.assertLess(packet["header"]["path_manager_plan_blend_alpha"], 0.25)
        self.assertLess(abs(packet["points"][1]["y_m"]), 0.5)
        self.assertTrue(debug["plan_blend_active"])

    def test_path_manager_stabilized_pose_clamps_yaw_spike(self) -> None:
        bridge = UdpResultBridge(UdpBridgeConfig(enabled=False, action_log_interval_s=-1.0))
        manager = LivePathManager(
            LivePathManagerConfig(
                enabled=True,
                stabilize_local_frame=True,
                yaw_filter_tau_s=0.5,
                yaw_max_rate_rad_s=0.2,
                log_interval_s=-1.0,
            ),
            udp_bridge=bridge,
            pose_source=FakePoseSource(),
        )
        try:
            t0_ns = time.time_ns()
            first = manager._stabilize_pose(make_pose(yaw_rad=0.0, receive_time_utc_ns=t0_ns))
            second = manager._stabilize_pose(make_pose(yaw_rad=1.0, receive_time_utc_ns=t0_ns + 100_000_000))
        finally:
            bridge.stop()

        self.assertAlmostEqual(first.yaw_rad, 0.0, places=5)
        self.assertLess(abs(second.yaw_rad), 0.01)

    def test_path_manager_stabilized_projection_arc_does_not_go_backward(self) -> None:
        bridge = UdpResultBridge(UdpBridgeConfig(enabled=False, action_log_interval_s=-1.0))
        pose_source = FakePoseSource()
        manager = LivePathManager(
            LivePathManagerConfig(
                enabled=True,
                max_plan_age_s=5.0,
                min_plan_arc_m=1.0,
                min_remaining_distance_m=1.0,
                max_projection_distance_m=5.0,
                stabilize_local_frame=True,
                projection_arc_filter_tau_s=0.5,
                log_interval_s=-1.0,
            ),
            udp_bridge=bridge,
            pose_source=pose_source,
        )
        try:
            info = manager.publish_result(make_result(sequence=21))
            self.assertTrue(info["path_manager_plan_accepted"])
            self.assertIsNotNone(manager._plan)
            assert manager._plan is not None
            packet_a, _ = manager._build_packet(manager._plan, make_pose(x_m=102.0, y_m=200.0))
            packet_b, _ = manager._build_packet(manager._plan, make_pose(x_m=101.0, y_m=200.0))
        finally:
            bridge.stop()

        self.assertIsNotNone(packet_a)
        self.assertIsNotNone(packet_b)
        assert packet_a is not None
        assert packet_b is not None
        arc_a = packet_a["header"]["path_manager_projection_arc_m"]
        arc_b = packet_b["header"]["path_manager_projection_arc_m"]
        self.assertGreaterEqual(arc_b + 1e-5, arc_a)

    def test_path_manager_fixed_arc_resampling_uses_uniform_spacing(self) -> None:
        bridge = UdpResultBridge(UdpBridgeConfig(enabled=False, action_log_interval_s=-1.0))
        pose_source = FakePoseSource()
        manager = LivePathManager(
            LivePathManagerConfig(
                enabled=True,
                max_plan_age_s=5.0,
                min_plan_arc_m=1.0,
                min_remaining_distance_m=1.0,
                max_projection_distance_m=5.0,
                output_points=8,
                stabilize_local_frame=True,
                fixed_arc_step_m=0.25,
                log_interval_s=-1.0,
            ),
            udp_bridge=bridge,
            pose_source=pose_source,
        )
        try:
            info = manager.publish_result(make_result(sequence=31))
            self.assertTrue(info["path_manager_plan_accepted"])
            self.assertIsNotNone(manager._plan)
            assert manager._plan is not None
            packet, _ = manager._build_packet(manager._plan, pose_source.pose)
        finally:
            bridge.stop()

        self.assertIsNotNone(packet)
        assert packet is not None
        xs = [float(point["x_m"]) for point in packet["points"][:5]]
        diffs = [xs[idx + 1] - xs[idx] for idx in range(len(xs) - 1)]
        for diff in diffs:
            self.assertAlmostEqual(diff, 0.25, places=4)
        self.assertAlmostEqual(packet["header"]["path_manager_fixed_arc_step_m"], 0.25, places=4)

    def test_path_manager_publishes_local_reorigin_packet(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(("127.0.0.1", 0))
        listener.settimeout(2.0)
        bridge = UdpResultBridge(
            UdpBridgeConfig(
                enabled=True,
                host="127.0.0.1",
                port=listener.getsockname()[1],
                payload_mode="text_json",
                action_log_interval_s=-1.0,
            )
        )
        pose_source = FakePoseSource()
        manager = LivePathManager(
            LivePathManagerConfig(
                enabled=True,
                rate_hz=20.0,
                max_plan_age_s=5.0,
                min_plan_arc_m=1.0,
                min_remaining_distance_m=1.0,
                output_points=8,
                log_interval_s=-1.0,
            ),
            udp_bridge=bridge,
            pose_source=pose_source,
        )
        try:
            manager.start()
            info = manager.publish_result(make_result())
            self.assertTrue(info["path_manager_plan_accepted"])
            data, _ = listener.recvfrom(65536)
        finally:
            manager.stop()
            bridge.stop()
            listener.close()

        payload = json.loads(data.decode("utf-8"))
        self.assertTrue(payload["packet_header"]["path_manager_enabled"])
        self.assertEqual(payload["packet_header"]["path_manager_mode"], "gnss_projection")
        self.assertEqual(payload["packet_header"]["num_points"], 8)
        self.assertAlmostEqual(payload["packet_points"][0]["x_m"], 0.0, places=4)
        self.assertAlmostEqual(payload["packet_points"][0]["y_m"], 0.0, places=4)
        self.assertGreater(payload["packet_header"]["path_manager_remaining_distance_m"], 1.0)


if __name__ == "__main__":
    unittest.main()
