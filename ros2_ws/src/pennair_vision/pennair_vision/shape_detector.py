#!/usr/bin/env python3
"""
Run the background-agnostic 3D shape detector on an incoming image stream.

    in   /camera/image_raw        sensor_msgs/Image
         /camera/camera_info      sensor_msgs/CameraInfo      (K)

    out  /shapes/detections       pennair_msgs/ShapeDetectionArray
         /shapes/detections_2d    vision_msgs/Detection2DArray
         /shapes/markers          visualization_msgs/MarkerArray
         /shapes/image_annotated  sensor_msgs/Image

The pipeline was already built for this. detect_video_3d.py runs under a strict
streaming contract -- one frame in at a time, `detect()` a pure function of that
frame, every piece of state living in the tracker and the scale memory -- which
is the same shape as a subscriber callback. So the per-frame body below is lifted
from `detect_video_3d.run()` unchanged, and the rest of this file is message
plumbing.

Two things ROS adds that the CLI could not demonstrate:

  * Frames really do arrive asynchronously, and get dropped when the detector
    cannot keep up. See the note on QoS in __init__ -- this is the correct
    behaviour for a live feed, not a shortfall.
  * Intrinsics arrive on a topic, so the detector is not told in advance what
    resolution or calibration it is working with.

Positions are published in METRES in the camera optical frame (REP-103). The
algorithm works in inches; the conversion happens at the publish boundary and
nowhere else.
"""

import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import (Detection2D, Detection2DArray,
                             ObjectHypothesisWithPose)
from visualization_msgs.msg import Marker, MarkerArray

from pennair_msgs.msg import ShapeDetection, ShapeDetectionArray
from pennair_vision import algo


class ShapeDetector(Node):

    def __init__(self):
        super().__init__("shape_detector")

        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("publish_markers", True)
        self.declare_parameter("publish_vision_msgs", True)
        self.declare_parameter("principal_point", "given")
        self.declare_parameter("k_width", float(algo.pose3d.K_REF_WIDTH))
        self.declare_parameter("marker_lifetime", 0.5)

        self.principal = self.get_parameter("principal_point").value
        self.k_width = float(self.get_parameter("k_width").value)

        # State. All of it lives here, none of it in the detector -- which is
        # what makes a bad frame unable to corrupt the frames after it.
        self.bridge = CvBridge()
        self.tracker = None
        self.camera = None
        self.plane = None
        self.info = None
        self.frame_idx = 0
        self.proc_ms = []

        # Best-effort, depth 1, on both ends. The 3D pipeline runs at roughly
        # 12 fps on a laptop and less in a VM, against a publisher that does not
        # wait for it. With this profile the node always works on the newest
        # frame and discards the backlog; with a reliable, deep queue it would
        # instead accumulate unbounded lag and report positions for a scene that
        # had already moved on. Dropping frames is what a drone does.
        self.create_subscription(Image, "/camera/image_raw",
                                 self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, "/camera/camera_info",
                                 self.on_info, qos_profile_sensor_data)

        self.det_pub = self.create_publisher(
            ShapeDetectionArray, "/shapes/detections", 10)
        self.vis_pub = self.create_publisher(
            Detection2DArray, "/shapes/detections_2d", 10)
        self.marker_pub = self.create_publisher(
            MarkerArray, "/shapes/markers", 10)
        self.image_pub = self.create_publisher(
            Image, "/shapes/image_annotated", qos_profile_sensor_data)

        self.get_logger().info("shape_detector ready, waiting for images")

    # ------------------------------------------------------------------
    # camera model
    # ------------------------------------------------------------------

    def on_info(self, msg):
        """Take K from the topic, once. Re-read only if the geometry changes."""
        if self.info is not None and (self.info.width, self.info.height,
                                      tuple(self.info.k)) == (msg.width, msg.height,
                                                              tuple(msg.k)):
            return
        self.info = msg
        # ref_width == the actual width, so `for_frame` scales by 1 and the K on
        # the wire is used exactly as published -- whoever produced it already
        # accounted for any resizing.
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.camera = algo.pose3d.Camera(
            K, ref_width=float(msg.width), principal="given"
        ).for_frame((msg.height, msg.width))
        self.plane = algo.pose3d.PlaneScale(self.camera)
        self.get_logger().info(
            f"camera from /camera/camera_info: {msg.width}x{msg.height} "
            f"fx={self.camera.fx:.2f} cx={self.camera.cx:.1f}")

    def _fallback_camera(self, frame):
        """No CameraInfo publisher? Fall back to the calibration in pose3d."""
        self.camera = algo.pose3d.Camera(
            ref_width=self.k_width, principal=self.principal
        ).for_frame(frame.shape)
        self.plane = algo.pose3d.PlaneScale(self.camera)
        self.get_logger().warn(
            "no /camera/camera_info seen; using the built-in calibration "
            f"(fx={self.camera.fx:.2f})")

    # ------------------------------------------------------------------
    # the pipeline
    # ------------------------------------------------------------------

    def on_image(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if self.camera is None:
            self._fallback_camera(frame)
        if self.tracker is None:
            self.tracker = algo.tracking.ShapeTracker(frame.shape)

        t0 = time.perf_counter()

        # --- lifted verbatim from detect_video_3d.run() -------------------
        detections, _, _ = algo.detector.detect(frame)   # stateless, this frame only
        tracks = self.tracker.update(detections, self.frame_idx)  # causal, past only

        obs = [(t.id, t.label, t.area, algo.video3d.measurable(t, frame.shape),
                algo.pose3d.circle_score(t.contour)) for t in tracks]
        Z, src = self.plane.update(obs)
        for t in tracks:
            t.xyz = self.plane.locate(t.center, Z) if Z else None
        # ------------------------------------------------------------------

        self.proc_ms.append((time.perf_counter() - t0) * 1000.0)
        del self.proc_ms[:-30]
        fps = 1000.0 / max(float(np.mean(self.proc_ms)), 1e-6)

        header = msg.header
        self.det_pub.publish(self._detection_array(header, tracks, Z, src))

        if self.get_parameter("publish_vision_msgs").value:
            self.vis_pub.publish(self._detection_2d_array(header, tracks))
        if self.get_parameter("publish_markers").value:
            self.marker_pub.publish(self._markers(header, tracks, Z))
        if self.get_parameter("publish_annotated").value:
            vis = algo.video3d.draw_overlay(frame, tracks, self.frame_idx, fps,
                                            detections, Z, src, "in")
            out = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            out.header = header
            self.image_pub.publish(out)

        self.frame_idx += 1
        if self.frame_idx % 50 == 0:
            depth = f"{Z * algo.pose3d.UNITS['m']:.2f} m [{src}]" if Z else "--"
            self.get_logger().info(
                f"frame {self.frame_idx}  {np.mean(self.proc_ms):5.1f} ms  "
                f"{fps:4.1f} fps  tracking {len(tracks)}  depth {depth}")

    # ------------------------------------------------------------------
    # message building
    # ------------------------------------------------------------------

    @staticmethod
    def _metres(xyz):
        return algo.pose3d.convert(xyz, "m")

    def _detection_array(self, header, tracks, Z, src):
        out = ShapeDetectionArray()
        out.header = header
        out.plane_depth = float(self._metres((0.0, 0.0, Z))[2]) if Z else 0.0
        out.depth_source = src or ""

        for t in tracks:
            d = ShapeDetection()
            d.track_id = int(t.id)
            d.shape = str(t.label)
            d.confidence = float(t.confidence)
            d.state = str(t.state)
            d.area_px = float(t.area)
            d.center_px = Point(x=float(t.center[0]), y=float(t.center[1]), z=0.0)

            if getattr(t, "xyz", None):
                X, Y, Zc = self._metres(t.xyz)
                d.position = Point(x=float(X), y=float(Y), z=float(Zc))
                d.has_position = True
            else:
                d.has_position = False

            # The refined contour itself, not an approximation of it -- this is
            # the outline the centre and the area were measured from.
            d.outline = [Point(x=float(p[0][0]), y=float(p[0][1]), z=0.0)
                         for p in t.contour]
            out.detections.append(d)
        return out

    def _detection_2d_array(self, header, tracks):
        """The same detections in a standard type, so off-the-shelf tools work."""
        out = Detection2DArray()
        out.header = header
        for t in tracks:
            x, y, w, h = cv2.boundingRect(t.contour)
            det = Detection2D()
            det.header = header
            det.id = str(t.id)
            det.bbox.center.position.x = float(x + w / 2.0)
            det.bbox.center.position.y = float(y + h / 2.0)
            det.bbox.center.theta = 0.0
            det.bbox.size_x = float(w)
            det.bbox.size_y = float(h)
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(t.label)
            hyp.hypothesis.score = float(t.confidence)
            det.results.append(hyp)
            out.detections.append(det)
        return out

    def _markers(self, header, tracks, Z):
        """RViz: a sphere at each centre, its label, and its outline in 3D.

        The outline is worth the extra few lines. Every contour point lies on the
        same plane at the same depth, so back-projecting it through the camera is
        exact -- the result is the shape's true outline in metres, not a billboard
        drawn at its centre.
        """
        out = MarkerArray()
        if not Z:
            return out

        lifetime = float(self.get_parameter("marker_lifetime").value)

        for t in tracks:
            if not getattr(t, "xyz", None):
                continue
            X, Y, Zc = self._metres(t.xyz)
            colour = ColorRGBA(r=float(t.color[2]) / 255.0,
                               g=float(t.color[1]) / 255.0,
                               b=float(t.color[0]) / 255.0, a=0.9)   # BGR -> RGB

            sphere = self._marker(header, colour, lifetime, t.id * 3, Marker.SPHERE)
            sphere.pose.position = Point(x=float(X), y=float(Y), z=float(Zc))
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.08
            out.markers.append(sphere)

            text = self._marker(header, colour, lifetime, t.id * 3 + 1,
                                Marker.TEXT_VIEW_FACING)
            text.pose.position = Point(x=float(X), y=float(Y),
                                       z=float(Zc) - 0.15)
            text.scale.z = 0.10
            text.text = f"#{t.id} {t.label}  {Zc:.2f} m"
            out.markers.append(text)

            line = self._marker(header, colour, lifetime, t.id * 3 + 2,
                                Marker.LINE_STRIP)
            line.scale.x = 0.01
            pts = [self._metres(self.camera.backproject(float(p[0][0]),
                                                        float(p[0][1]), Z))
                   for p in t.contour]
            line.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
                           for p in pts]
            if line.points:
                line.points.append(line.points[0])          # close the outline
            out.markers.append(line)
        return out

    @staticmethod
    def _marker(header, colour, lifetime, marker_id, marker_type):
        """A marker with everything common already filled in.

        `lifetime` rather than a DELETEALL each frame: a track that disappears
        takes its marker with it after a beat, and RViz never sees the flicker a
        clear-then-redraw produces.
        """
        m = Marker()
        m.header = header
        m.ns = "shapes"
        m.action = Marker.ADD
        m.id = int(marker_id)
        m.type = marker_type
        m.color = colour
        m.pose.orientation.w = 1.0
        m.lifetime.sec = int(lifetime)
        m.lifetime.nanosec = int((lifetime - int(lifetime)) * 1e9)
        return m


def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetector()
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
