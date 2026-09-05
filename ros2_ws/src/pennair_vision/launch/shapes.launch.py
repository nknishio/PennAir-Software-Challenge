#!/usr/bin/env python3
"""
Bring up the whole system: video source, detector, TF, and optionally RViz.

    ros2 launch pennair_vision shapes.launch.py \
        video:=$HOME/pennair/"PennAir 2024 App Dynamic Hard.mp4" \
        principal_point:=center rviz:=true

The one job that is not obvious is PYTHONPATH. The detection algorithm lives at
the repository root and is deliberately left there -- unchanged, so that
`python3 run_tests.py` and every command in the README keep working. This launch
file points the nodes at it via PENNAIR_ROOT rather than the package vendoring a
second copy of code that already passes its own test suite.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, SetEnvironmentVariable)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def _default_repo_root():
    """The repo root, as seen from a source checkout of this package."""
    env = os.environ.get("PENNAIR_ROOT")
    if env:
        return env
    here = os.path.abspath(__file__)
    for _ in range(6):
        here = os.path.dirname(here)
        if os.path.isfile(os.path.join(here, "pose3d.py")):
            return here
    return os.path.expanduser("~/pennair")


def generate_launch_description():
    share = get_package_share_directory("pennair_vision")

    args = [
        DeclareLaunchArgument(
            "video", default_value="",
            description="Video file to stream, or '0' for a webcam"),
        DeclareLaunchArgument(
            "repo_root", default_value=_default_repo_root(),
            description="Repository root holding pose3d.py and the detectors"),
        DeclareLaunchArgument("rate", default_value="10.0",
                              description="Publish rate in Hz"),
        DeclareLaunchArgument("scale", default_value="0.5",
                              description="Frame downscale factor"),
        DeclareLaunchArgument("loop", default_value="true"),
        DeclareLaunchArgument(
            "principal_point", default_value="given",
            description="'given' uses K as supplied (cx=cy=0); 'center' "
                        "recentres it, which makes the RViz view look natural"),
        DeclareLaunchArgument("rviz", default_value="false"),
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution([share, "config", "params.yaml"])),
    ]

    repo_root = LaunchConfiguration("repo_root")
    params = LaunchConfiguration("params_file")
    principal = LaunchConfiguration("principal_point")

    # Both nodes resolve the algorithm through pennair_vision.algo, which reads
    # this. Setting PYTHONPATH as well would work, but this is narrower: it says
    # where *this* project lives rather than editing a global search path.
    set_root = SetEnvironmentVariable("PENNAIR_ROOT", repo_root)

    video_publisher = Node(
        package="pennair_vision",
        executable="video_publisher",
        name="video_publisher",
        output="screen",
        parameters=[params, {
            "video_path": LaunchConfiguration("video"),
            "rate": LaunchConfiguration("rate"),
            "scale": LaunchConfiguration("scale"),
            "loop": LaunchConfiguration("loop"),
            "principal_point": principal,
        }],
    )

    shape_detector = Node(
        package="pennair_vision",
        executable="shape_detector",
        name="shape_detector",
        output="screen",
        parameters=[params, {"principal_point": principal}],
    )

    # camera_link (x forward, y left, z up) -> camera_optical_frame
    # (x right, y down, z forward). The standard REP-103 optical rotation. The
    # pinhole model already produces optical-frame coordinates, so the detector
    # needs no axis remapping; this exists so the markers sit correctly if a
    # vehicle frame is ever added above them.
    optical_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_optical_tf",
        arguments=["--x", "0", "--y", "0", "--z", "0",
                   "--roll", "-1.5707963", "--pitch", "0", "--yaw", "-1.5707963",
                   "--frame-id", "camera_link",
                   "--child-frame-id", "camera_optical_frame"],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", PathJoinSubstitution([share, "rviz", "shapes.rviz"])],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(args + [set_root, video_publisher, shape_detector,
                                     optical_tf, rviz])
