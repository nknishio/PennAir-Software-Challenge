"""
Locate and import the detection algorithm, which lives outside this package.

The algorithm modules stay at the repository root, unchanged and unrestructured,
so that `python3 run_tests.py` and every command in the README keep working
exactly as documented. This ROS package is a layer over them, not a fork of them,
and nothing here may modify them.

That leaves one job: making `import pose3d` work from inside a ROS node. Three
ways are tried, in order of how explicit they are:

  1. PENNAIR_ROOT in the environment -- what the launch file sets.
  2. Already importable, because PYTHONPATH was set by hand.
  3. Walk up from this file looking for pose3d.py -- true in a source checkout,
     which is how `colcon build --symlink-install` leaves things.

If all three fail the error says so plainly, because "ModuleNotFoundError:
pose3d" three frames deep inside rclpy is a genuinely confusing thing to hit.
"""

import os
import sys

_MARKER = "pose3d.py"


def _candidate_roots():
    root = os.environ.get("PENNAIR_ROOT")
    if root:
        yield os.path.abspath(os.path.expanduser(root))
    # ros2_ws/src/pennair_vision/pennair_vision/algo.py -> repo root is 5 up
    here = os.path.abspath(__file__)
    for _ in range(6):
        here = os.path.dirname(here)
        yield here


def _resolve():
    try:                                    # already on PYTHONPATH?
        import pose3d                       # noqa: F401
        return None
    except ImportError:
        pass

    for root in _candidate_roots():
        if root and os.path.isfile(os.path.join(root, _MARKER)):
            sys.path.insert(0, root)
            return root

    raise ImportError(
        "Could not find the PennAir detection modules (looking for "
        f"{_MARKER}). Set PENNAIR_ROOT to the repository root, e.g.\n"
        "    export PENNAIR_ROOT=$HOME/pennair\n"
        "or pass repo_root:=... to the launch file."
    )


REPO_ROOT = _resolve()

# Imported after the path is fixed, and re-exported so the nodes have one place
# to import from.
import detect_shapes_agnostic as detector          # noqa: E402
import detect_video_agnostic as tracking           # noqa: E402
import detect_video_3d as video3d                  # noqa: E402
import pose3d                                      # noqa: E402

__all__ = ["detector", "tracking", "video3d", "pose3d", "REPO_ROOT"]
