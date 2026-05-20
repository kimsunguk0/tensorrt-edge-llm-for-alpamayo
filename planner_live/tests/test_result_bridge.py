from __future__ import annotations

import socket
import time
import unittest
import json

from scripts.control_team_replay_common import unpack_packet

from planner_live.result_bridge import (
    UdpBridgeConfig,
    UdpResultBridge,
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
        self.assertEqual(payload["udp_mode"], "text_json_live")
        self.assertAlmostEqual(payload["actual_offset_s"], 0.15, places=6)
        self.assertAlmostEqual(payload["target_offset_s"], 0.15, places=6)

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
