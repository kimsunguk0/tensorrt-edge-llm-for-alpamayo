from __future__ import annotations

import unittest

from planner_live.planner_live_service import LatestOnlyBuffer


class LatestOnlyBufferTests(unittest.TestCase):
    def test_replaces_pending_item(self) -> None:
        queue: LatestOnlyBuffer[str] = LatestOnlyBuffer()
        self.assertIsNone(queue.offer("sample-1"))
        self.assertEqual(queue.offer("sample-2"), "sample-1")
        self.assertEqual(queue.take(timeout=0.01), "sample-2")

    def test_close_unblocks_take(self) -> None:
        queue: LatestOnlyBuffer[str] = LatestOnlyBuffer()
        queue.close()
        self.assertIsNone(queue.take(timeout=0.01))

    def test_discard_if_removes_matching_pending_item(self) -> None:
        queue: LatestOnlyBuffer[int] = LatestOnlyBuffer()
        queue.offer(10)
        self.assertEqual(queue.discard_if(lambda value: value <= 10), 10)
        self.assertIsNone(queue.take(timeout=0.01))

    def test_discard_if_keeps_non_matching_pending_item(self) -> None:
        queue: LatestOnlyBuffer[int] = LatestOnlyBuffer()
        queue.offer(11)
        self.assertIsNone(queue.discard_if(lambda value: value <= 10))
        self.assertEqual(queue.take(timeout=0.01), 11)


if __name__ == "__main__":
    unittest.main()
