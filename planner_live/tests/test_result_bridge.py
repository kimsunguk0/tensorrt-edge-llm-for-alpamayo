from __future__ import annotations

import socket
import time
import unittest
import json
import tempfile
from pathlib import Path

from scripts.control_team_replay_common import unpack_packet

from planner_live.result_bridge import (
    UdpBridgeConfig,
    UdpResultBridge,
    blend_with_previous_full_plan_packet,
    build_full_result_packet,
    build_live_result_packet,
    build_text_result_payload,
)
from planner_live.result_parser import PlannerResult


def _identity_rot() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def make_result() -> PlannerResult:
    return PlannerResult(
        sequence=42,
        t0_us=1_000_000,
        clip_id="clip",
        completed_at_unix=time.time(),
        output_text="keep lane",
        fm_mode="single_branch",
        fm_status="completed",
        plan_dt_s=0.1,
        pred_xyz=[
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.3, 0.0, 0.0],
        ],
        pred_rot=[_identity_rot(), _identity_rot(), _identity_rot()],
        post_vlm_timing={},
        fm_timing={},
        output_json_path="/tmp/output.json",
    )


def make_result_with_lateral_y(y_m: float, *, sequence: int = 42) -> PlannerResult:
    result = make_result()
    result.sequence = sequence
    result.pred_xyz = [
        [0.1, y_m, 0.0],
        [0.2, y_m, 0.0],
        [0.3, y_m, 0.0],
    ]
    return result


class ResultBridgeTests(unittest.TestCase):
    def test_build_live_result_packet_resamples_from_plan_age(self) -> None:
        packet = build_live_result_packet(
            make_result(),
            tx_seq=7,
            tx_time_us=1_150_000,
            control_dt_s=0.05,
            control_points=4,
        )
        self.assertEqual(packet["header"]["tx_seq"], 7)
        self.assertEqual(packet["header"]["plan_seq"], 42)
        self.assertEqual(packet["header"]["sample_id"], 42)
        self.assertEqual(packet["header"]["num_points"], 4)
        xs = [point["x_m"] for point in packet["points"]]
        self.assertAlmostEqual(xs[0], 0.15, places=5)
        self.assertAlmostEqual(xs[1], 0.20, places=5)
        self.assertAlmostEqual(xs[2], 0.25, places=5)
        self.assertAlmostEqual(xs[3], 0.30, places=5)

    def test_build_full_result_packet_keeps_entire_plan_from_origin(self) -> None:
        packet = build_full_result_packet(
            make_result(),
            tx_seq=7,
            tx_time_us=1_150_000,
        )
        self.assertEqual(packet["header"]["tx_seq"], 7)
        self.assertEqual(packet["header"]["plan_seq"], 42)
        self.assertEqual(packet["header"]["sample_id"], 42)
        self.assertEqual(packet["header"]["num_points"], 4)
        self.assertAlmostEqual(packet["header"]["dt_s"], 0.1, places=5)
        xs = [point["x_m"] for point in packet["points"]]
        self.assertAlmostEqual(xs[0], 0.0, places=5)
        self.assertAlmostEqual(xs[1], 0.1, places=5)
        self.assertAlmostEqual(xs[2], 0.2, places=5)
        self.assertAlmostEqual(xs[3], 0.3, places=5)

    def test_build_full_result_packet_can_latency_compensate_and_reorigin(self) -> None:
        packet = build_full_result_packet(
            make_result(),
            tx_seq=7,
            tx_time_us=1_150_000,
            latency_compensate=True,
        )
        self.assertTrue(packet["header"]["latency_compensated_full_plan"])
        self.assertAlmostEqual(packet["header"]["latency_compensation_age_s"], 0.15, places=5)
        self.assertEqual(packet["header"]["num_points"], 4)
        xs = [point["x_m"] for point in packet["points"]]
        ys = [point["y_m"] for point in packet["points"]]
        self.assertAlmostEqual(xs[0], 0.0, places=5)
        self.assertAlmostEqual(xs[1], 0.10, places=5)
        self.assertAlmostEqual(xs[2], 0.15, places=5)
        self.assertAlmostEqual(xs[3], 0.15, places=5)
        self.assertTrue(all(abs(y) < 1e-6 for y in ys))

    def test_previous_full_plan_blend_shifts_and_reorigins_previous_path(self) -> None:
        previous_packet = build_full_result_packet(
            make_result_with_lateral_y(0.0, sequence=41),
            tx_seq=1,
            tx_time_us=1_000_000,
        )
        current_packet = build_full_result_packet(
            make_result_with_lateral_y(1.0, sequence=42),
            tx_seq=2,
            tx_time_us=1_100_000,
        )

        blended = blend_with_previous_full_plan_packet(
            current_packet,
            previous_packet,
            ratio=0.5,
            max_age_s=3.0,
        )

        self.assertTrue(blended["header"]["previous_path_blend_applied"])
        self.assertEqual(blended["header"]["previous_path_blend_shift_points"], 1)
        self.assertEqual(blended["header"]["previous_path_blend_previous_plan_seq"], 41)
        self.assertAlmostEqual(blended["points"][0]["x_m"], 0.0, places=5)
        self.assertAlmostEqual(blended["points"][0]["y_m"], 0.0, places=5)
        self.assertAlmostEqual(blended["points"][1]["x_m"], 0.1, places=5)
        self.assertAlmostEqual(blended["points"][1]["y_m"], 0.5, places=5)

    def test_udp_bridge_sends_packet(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(("127.0.0.1", 0))
        listener.settimeout(2.0)
        bridge = UdpResultBridge(
            UdpBridgeConfig(
                enabled=True,
                host="127.0.0.1",
                port=listener.getsockname()[1],
                rate_hz=20.0,
                control_dt_s=0.05,
                control_points=4,
            )
        )
        try:
            bridge.start()
            bridge.publish_result(make_result())
            data, _ = listener.recvfrom(4096)
        finally:
            bridge.stop()
            listener.close()

        packet = unpack_packet(data)
        self.assertEqual(packet["header"]["plan_seq"], 42)
        self.assertEqual(packet["header"]["sample_id"], 42)
        self.assertEqual(packet["header"]["num_points"], 4)

    def test_udp_bridge_full_plan_mode_sends_full_packet(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(("127.0.0.1", 0))
        listener.settimeout(2.0)
        bridge = UdpResultBridge(
            UdpBridgeConfig(
                enabled=True,
                host="127.0.0.1",
                port=listener.getsockname()[1],
                rate_hz=20.0,
                control_dt_s=0.05,
                control_points=4,
                full_plan=True,
            )
        )
        try:
            bridge.start()
            bridge.publish_result(make_result())
            data, _ = listener.recvfrom(4096)
        finally:
            bridge.stop()
            listener.close()

        packet = unpack_packet(data)
        self.assertEqual(packet["header"]["plan_seq"], 42)
        self.assertEqual(packet["header"]["sample_id"], 42)
        self.assertEqual(packet["header"]["num_points"], 4)
        self.assertAlmostEqual(packet["header"]["dt_s"], 0.1, places=5)
        xs = [point["x_m"] for point in packet["points"]]
        self.assertAlmostEqual(xs[0], 0.0, places=5)
        self.assertAlmostEqual(xs[-1], 0.3, places=5)

    def test_udp_bridge_send_result_once(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(("127.0.0.1", 0))
        listener.settimeout(2.0)
        bridge = UdpResultBridge(
            UdpBridgeConfig(
                enabled=True,
                host="127.0.0.1",
                port=listener.getsockname()[1],
                rate_hz=20.0,
                control_dt_s=0.05,
                control_points=4,
                full_plan=True,
            )
        )
        try:
            info = bridge.send_result_once(make_result())
            data, _ = listener.recvfrom(4096)
        finally:
            bridge.stop()
            listener.close()

        self.assertTrue(info["udp_sent"])
        packet = unpack_packet(data)
        self.assertEqual(packet["header"]["plan_seq"], 42)
        self.assertEqual(packet["header"]["sample_id"], 42)

    def test_text_json_payload_matches_viewer_shape(self) -> None:
        packet = build_full_result_packet(make_result(), tx_seq=3, tx_time_us=1_150_000)
        payload = build_text_result_payload(make_result(), packet)
        self.assertEqual(payload["packet_header"]["magic"], "ALPA")
        self.assertEqual(payload["packet_header"]["num_points"], 4)
        self.assertEqual(len(payload["packet_points"]), 4)
        self.assertEqual(payload["pred_xyz_source"], "packet_points_xy")
        self.assertTrue(payload["packet_points_include_origin"])
        self.assertEqual(len(payload["pred_xyz"]), 3)
        self.assertAlmostEqual(payload["pred_xyz"][0][0], 0.1, places=5)
        self.assertEqual(payload["pred_xyz"][0][1:], [0.0, 0.0])
        self.assertEqual(payload["udp_mode"], "text_json_live")
        self.assertAlmostEqual(payload["actual_offset_s"], 0.15, places=6)
        self.assertAlmostEqual(payload["target_offset_s"], 0.15, places=6)

    def test_text_json_payload_pred_xyz_uses_live_packet_path_xy(self) -> None:
        packet = build_live_result_packet(
            make_result(),
            tx_seq=3,
            tx_time_us=1_150_000,
            control_dt_s=0.05,
            control_points=4,
        )
        payload = build_text_result_payload(make_result(), packet)
        self.assertFalse(payload["packet_points_include_origin"])
        self.assertEqual(len(payload["pred_xyz"]), 4)
        self.assertAlmostEqual(payload["pred_xyz"][0][0], 0.15, places=5)
        self.assertAlmostEqual(payload["pred_xyz"][1][0], 0.20, places=5)
        self.assertEqual(payload["pred_xyz"][0][1], 0.0)

    def test_udp_bridge_text_json_mode_sends_json(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(("127.0.0.1", 0))
        listener.settimeout(2.0)
        bridge = UdpResultBridge(
            UdpBridgeConfig(
                enabled=True,
                host="127.0.0.1",
                port=listener.getsockname()[1],
                payload_mode="text_json",
                rate_hz=20.0,
                full_plan=True,
            )
        )
        try:
            info = bridge.send_result_once(make_result())
            data, _ = listener.recvfrom(4096)
        finally:
            bridge.stop()
            listener.close()

        self.assertTrue(info["udp_sent"])
        payload = json.loads(data.decode("utf-8"))
        self.assertEqual(payload["packet_header"]["plan_seq"], 42)
        self.assertEqual(len(payload["packet_points"]), 4)
        self.assertGreater(payload["actual_offset_s"], 0.0)
        self.assertAlmostEqual(payload["target_offset_s"], payload["actual_offset_s"], places=6)

    def test_udp_bridge_text_json_full_plan_latency_compensation(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(("127.0.0.1", 0))
        listener.settimeout(2.0)
        result = make_result()
        result.t0_us = time.time_ns() // 1000 - 150_000
        bridge = UdpResultBridge(
            UdpBridgeConfig(
                enabled=True,
                host="127.0.0.1",
                port=listener.getsockname()[1],
                payload_mode="text_json",
                send_mode="on_result",
                full_plan=True,
                latency_compensate_full_plan=True,
            )
        )
        try:
            info = bridge.send_result_once(result)
            data, _ = listener.recvfrom(4096)
        finally:
            bridge.stop()
            listener.close()

        self.assertTrue(info["udp_sent"])
        payload = json.loads(data.decode("utf-8"))
        self.assertTrue(payload["packet_header"]["latency_compensated_full_plan"])
        self.assertGreater(payload["packet_header"]["latency_compensation_age_s"], 0.0)
        self.assertAlmostEqual(payload["packet_points"][0]["x_m"], 0.0, places=5)
        self.assertAlmostEqual(payload["packet_points"][0]["y_m"], 0.0, places=5)

    def test_udp_bridge_saves_sent_path_log(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(("127.0.0.1", 0))
        listener.settimeout(2.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            path_log_dir = Path(tmpdir)
            bridge = UdpResultBridge(
                UdpBridgeConfig(
                    enabled=True,
                    host="127.0.0.1",
                    port=listener.getsockname()[1],
                    payload_mode="text_json",
                    send_mode="on_result",
                    full_plan=True,
                    path_log_dir=path_log_dir,
                )
            )
            try:
                info = bridge.send_result_once(make_result())
                listener.recvfrom(4096)
            finally:
                bridge.stop()
                listener.close()

            jsonl_path = path_log_dir / "udp_sent_paths.jsonl"
            latest_path = path_log_dir / "latest_udp_path.json"
            self.assertEqual(info["path_log_path"], str(jsonl_path))
            self.assertTrue(jsonl_path.exists())
            self.assertTrue(latest_path.exists())
            line = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            self.assertEqual(line["log_type"], "udp_sent_path")
            self.assertEqual(line["packet_header"]["plan_seq"], 42)
            self.assertEqual(len(line["packet_points"]), 4)
            self.assertEqual(line["target"]["host"], "127.0.0.1")
            self.assertFalse(line["gt_future_available_at_send_time"])
            self.assertEqual(latest["packet_header"]["tx_seq"], line["packet_header"]["tx_seq"])

    def test_udp_bridge_on_result_mode_sends_once_on_publish(self) -> None:
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(("127.0.0.1", 0))
        listener.settimeout(2.0)
        bridge = UdpResultBridge(
            UdpBridgeConfig(
                enabled=True,
                host="127.0.0.1",
                port=listener.getsockname()[1],
                payload_mode="text_json",
                send_mode="on_result",
                full_plan=True,
            )
        )
        try:
            bridge.start()
            bridge.publish_result(make_result())
            data, _ = listener.recvfrom(4096)
        finally:
            bridge.stop()
            listener.close()

        payload = json.loads(data.decode("utf-8"))
        self.assertEqual(payload["packet_header"]["plan_seq"], 42)
        self.assertEqual(payload["packet_header"]["num_points"], 4)


if __name__ == "__main__":
    unittest.main()
