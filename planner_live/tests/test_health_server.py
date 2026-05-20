from __future__ import annotations

import base64
import json
from pathlib import Path
import tempfile
import unittest
import urllib.request

from planner_live.health_server import HealthServer


PNG_1X1_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9sY9l1kAAAAASUVORK5CYII="
)


class HealthServerTests(unittest.TestCase):
    def test_viewer_and_artifact_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            latest_result = tmp_path / "latest_result.json"
            latest_result.write_text(json.dumps({"sequence": 7, "output_text": "keep lane"}))
            latest_trajectory = tmp_path / "latest_trajectory.json"
            latest_trajectory.write_text(json.dumps({"pred_xyz": [[0.0, 0.0, 0.0]]}))
            latest_dashboard = tmp_path / "latest_dashboard.png"
            latest_dashboard.write_bytes(base64.b64decode(PNG_1X1_BASE64))

            server = HealthServer(
                "127.0.0.1",
                0,
                lambda: {
                    "service_running": True,
                    "runtime_alive": True,
                    "inference_busy": False,
                    "manual_run_busy": False,
                    "manual_run_count": 0,
                    "manual_last_status": None,
                    "last_success_sequence": 7,
                },
                lambda: {
                    "latest_dashboard": latest_dashboard,
                    "latest_result": latest_result,
                    "latest_trajectory": latest_trajectory,
                },
                lambda: {"ok": True, "sequence": 7},
            )
            server.start()
            try:
                base_url = f"http://127.0.0.1:{server.port}"

                with urllib.request.urlopen(base_url + "/viewer", timeout=5) as response:
                    viewer_html = response.read().decode("utf-8")
                    self.assertEqual(response.status, 200)
                    self.assertIn("Planner Live Viewer", viewer_html)
                    self.assertIn("/artifacts/latest_dashboard.png", viewer_html)
                    self.assertIn("Final Output", viewer_html)

                with urllib.request.urlopen(base_url + "/viewer-manual", timeout=5) as response:
                    viewer_html = response.read().decode("utf-8")
                    self.assertEqual(response.status, 200)
                    self.assertIn("Planner Manual Viewer", viewer_html)
                    self.assertIn("Generate Route Once", viewer_html)

                with urllib.request.urlopen(base_url + "/artifacts/latest_result.json", timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    self.assertEqual(response.status, 200)
                    self.assertEqual(payload["sequence"], 7)
                    self.assertEqual(payload["output_text"], "keep lane")

                with urllib.request.urlopen(base_url + "/artifacts/latest_dashboard.png", timeout=5) as response:
                    body = response.read()
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.headers.get_content_type(), "image/png")
                    self.assertGreater(len(body), 0)

                request = urllib.request.Request(base_url + "/actions/manual-run-once", method="POST", data=b"{}")
                with urllib.request.urlopen(request, timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    self.assertEqual(response.status, 200)
                    self.assertTrue(payload["ok"])
                    self.assertEqual(payload["sequence"], 7)
            finally:
                server.stop()


if __name__ == "__main__":
    unittest.main()
