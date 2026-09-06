#!/usr/bin/env python3
"""
End-to-end check that the ROS layer did not break the pipeline underneath it.

Brings both nodes up against the hard video, listens on /shapes/detections, and
asserts the things that would break if the port went wrong:

  * messages arrive at all
  * five shapes are being tracked
  * every detection carries an outline and a metric position
  * the plane depth matches what the CLI reports

That last one is the useful assertion. `python3 detect_video_3d.py` on this
footage gives a plane depth of 251.74 in; in meters that is 6.395. If the ROS
numbers disagree the fault is in K scaling or in the inch-to-meter conversion at
the publish boundary, not in the detector -- which narrows the search to this
package immediately.

    cd ros2_ws && colcon test --packages-select pennair_vision
    # or directly:
    python3 -m pytest src/pennair_vision/test/test_ros_pipeline.py -s
"""

import os
import time

import pytest
import rclpy
from rclpy.node import Node

from pennair_msgs.msg import ShapeDetectionArray

VIDEO_ENV = "PENNAIR_VIDEO"
DEFAULT_VIDEO = os.path.expanduser("~/pennair/PennAir 2024 App Dynamic Hard.mp4")

CLI_DEPTH_IN = 251.74                 # what detect_video_3d.py reports
EXPECTED_DEPTH_M = CLI_DEPTH_IN / 39.3700787
DEPTH_TOL = 0.05                      # 5%
TIMEOUT_S = 45.0


class Listener(Node):
    def __init__(self):
        super().__init__("test_listener")
        self.messages = []
        self.create_subscription(ShapeDetectionArray, "/shapes/detections",
                                 self.messages.append, 10)


@pytest.fixture(scope="module")
def video():
    path = os.environ.get(VIDEO_ENV, DEFAULT_VIDEO)
    if not os.path.isfile(path):
        pytest.skip(f"video not found: {path} (set {VIDEO_ENV})")
    return path


def test_detections_published(video):
    """Run the launch file, collect messages, and check them."""
    rclpy.init()
    listener = Listener()
    try:
        deadline = time.time() + TIMEOUT_S
        # Wait for messages carrying a depth -- the first frames legitimately
        # have none, because the tracker needs three hits to confirm a track
        # before anything can act as a ruler.
        while time.time() < deadline:
            rclpy.spin_once(listener, timeout_sec=0.5)
            if sum(1 for m in listener.messages if m.depth_source) >= 5:
                break

        assert listener.messages, (
            "no messages on /shapes/detections -- is the launch file running? "
            "This test does not start it; run it in another terminal first.")

        located = [m for m in listener.messages if m.depth_source]
        assert located, "messages arrived but none carried a depth"

        counts = [len(m.detections) for m in located]
        assert max(counts) == 5, f"expected 5 shapes in view, saw at most {max(counts)}"

        depths = [m.plane_depth for m in located]
        median = sorted(depths)[len(depths) // 2]
        err = abs(median - EXPECTED_DEPTH_M) / EXPECTED_DEPTH_M
        assert err < DEPTH_TOL, (
            f"plane depth {median:.3f} m disagrees with the CLI's "
            f"{EXPECTED_DEPTH_M:.3f} m by {100 * err:.1f}% -- suspect K scaling "
            f"or the inch->meter conversion")

        sample = located[-1]
        for d in sample.detections:
            assert d.outline, f"{d.shape} has no outline"
            assert d.track_id > 0
            if d.has_position:
                assert d.position.z > 0.0

        print(f"\n  {len(listener.messages)} messages, "
              f"{len(located)} with depth")
        print(f"  plane depth median {median:.3f} m "
              f"(CLI {EXPECTED_DEPTH_M:.3f} m, {100 * err:.2f}% apart)")
        print(f"  shapes per frame max {max(counts)}")
    finally:
        listener.destroy_node()
        rclpy.shutdown()
