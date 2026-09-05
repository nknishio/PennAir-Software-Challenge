#!/usr/bin/env python3
"""
Stream a video file (or a live camera) onto ROS topics as if it were a drone feed.

Publishes two things, and the second one matters more than it looks:

    /camera/image_raw     sensor_msgs/Image        the frame
    /camera/camera_info   sensor_msgs/CameraInfo   the intrinsics, K

Shipping K on a topic rather than hardcoding it in the detector is the idiomatic
ROS arrangement, and here it also removes a specific way to be silently wrong.
`scale` downsamples frames so a VM can keep up with the bandwidth, and intrinsics
are measured *in pixels* -- so a resized image needs a resized K. pose3d's
`Camera.for_frame` already does exactly that, and this node calls it on the
frame it is actually about to publish. Without it, `scale:=0.5` would report
every shape at twice its true distance and nothing would look obviously broken.

Usage:
    ros2 run pennair_vision video_publisher --ros-args -p video_path:=clip.mp4
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from pennair_vision import algo


class VideoPublisher(Node):

    def __init__(self):
        super().__init__("video_publisher")

        self.declare_parameter("video_path", "")
        self.declare_parameter("rate", 10.0)
        self.declare_parameter("scale", 0.5)
        self.declare_parameter("loop", True)
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.declare_parameter("principal_point", "given")
        self.declare_parameter("k_width", float(algo.pose3d.K_REF_WIDTH))

        source = self.get_parameter("video_path").value
        self.rate = float(self.get_parameter("rate").value)
        self.scale = float(self.get_parameter("scale").value)
        self.loop = bool(self.get_parameter("loop").value)
        self.frame_id = self.get_parameter("frame_id").value

        if not source:
            raise RuntimeError("video_path is required (a file, or '0' for a webcam)")
        # An all-digits string means a camera index, which is how the CLI tools
        # spell it too -- so `video_path:=0` runs a live feed through this node.
        self.source = int(source) if str(source).isdigit() else source

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open {self.source}")

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, "/camera/image_raw",
                                               qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo, "/camera/camera_info",
                                              qos_profile_sensor_data)

        # Built lazily from the first frame, because K depends on the size of the
        # image actually being published, not on the size of the file.
        self.camera = None
        self.info = None
        self.frames = 0

        self.timer = self.create_timer(1.0 / max(self.rate, 0.1), self.tick)
        self.get_logger().info(
            f"streaming {self.source} at {self.rate} Hz, scale {self.scale}")

    # ------------------------------------------------------------------

    def _build_camera_info(self, frame):
        h, w = frame.shape[:2]
        cam = algo.pose3d.Camera(
            ref_width=float(self.get_parameter("k_width").value),
            principal=self.get_parameter("principal_point").value,
        ).for_frame(frame.shape)

        info = CameraInfo()
        info.height, info.width = h, w
        info.distortion_model = "plumb_bob"
        # K is supplied without distortion coefficients, so none are claimed.
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = [cam.fx, 0.0, cam.cx,
                  0.0, cam.fy, cam.cy,
                  0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        info.p = [cam.fx, 0.0, cam.cx, 0.0,
                  0.0, cam.fy, cam.cy, 0.0,
                  0.0, 0.0, 1.0, 0.0]
        self.get_logger().info(
            f"camera info for {w}x{h}: fx={cam.fx:.2f} fy={cam.fy:.2f} "
            f"cx={cam.cx:.1f} cy={cam.cy:.1f}")
        return cam, info

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            if not self.loop:
                self.get_logger().info("end of stream")
                self.timer.cancel()
                return
            # Rewind. This is a seek in the *source*, which a looping test
            # harness is entitled to do; the streaming contract constrains the
            # detector, which still sees nothing but a sequence of frames.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                self.get_logger().error("could not rewind the source")
                self.timer.cancel()
                return

        if self.scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_AREA)

        if self.info is None:
            self.camera, self.info = self._build_camera_info(frame)

        stamp = self.get_clock().now().to_msg()
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id

        self.info.header.stamp = stamp
        self.info.header.frame_id = self.frame_id

        self.image_pub.publish(msg)
        self.info_pub.publish(self.info)
        self.frames += 1

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
