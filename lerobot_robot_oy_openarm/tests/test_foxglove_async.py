#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import threading
import unittest

from lerobot_robot_oy_openarm.foxglove_async import _LatestFoxgloveDispatcher


class LatestFoxgloveDispatcherTest(unittest.TestCase):
    def test_does_not_block_and_keeps_latest_frame(self) -> None:
        started = threading.Event()
        release = threading.Event()
        calls: list[tuple[int, bool]] = []

        def slow_logger(
            display_mode: str,
            *,
            observation: dict,
            action: dict,
            compress_images: bool,
        ) -> None:
            self.assertEqual(display_mode, "foxglove")
            calls.append((observation["frame"], compress_images))
            started.set()
            release.wait(timeout=2)

        dispatcher = _LatestFoxgloveDispatcher(slow_logger)
        dispatcher.submit({"frame": 0}, {}, True)
        self.assertTrue(started.wait(timeout=1))

        # These submissions must replace the one pending item instead of
        # waiting for the slow renderer.
        dispatcher.submit({"frame": 1}, {}, False)
        dispatcher.submit({"frame": 2}, {}, False)
        release.set()
        dispatcher.close()

        self.assertEqual(calls, [(0, True), (2, True)])

    def test_failure_does_not_escape_record_loop(self) -> None:
        failed = threading.Event()

        def broken_logger(*args, **kwargs) -> None:
            failed.set()
            raise RuntimeError("render failed")

        dispatcher = _LatestFoxgloveDispatcher(broken_logger)
        dispatcher.submit({"frame": 0}, {}, True)
        self.assertTrue(failed.wait(timeout=1))
        dispatcher.close()


if __name__ == "__main__":
    unittest.main()

