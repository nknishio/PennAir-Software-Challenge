# Every shape found in one frame, plus the depth of the plane they lie on.
#
# The flat-surface assumption means a single depth describes the whole scene, so
# it belongs on the frame rather than repeated per shape.

std_msgs/Header header      # stamp and frame_id copied from the source image

float32 plane_depth         # meterS; 0.0 when no scale has been established yet

# How plane_depth was obtained on this frame:
#   circle   -- measured directly from the 10 in reference circle
#   learned  -- carried by a shape the circle had previously sized
#   held     -- nothing usable in view; the last known depth is being reported
#   ""       -- no scale yet
string depth_source

ShapeDetection[] detections
