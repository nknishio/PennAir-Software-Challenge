# One detected shape, as tracked through time.
#
# Carries what a bounding box cannot: the actual outline, a persistent identity,
# a metric position, and -- because the depth is inferred rather than sensed --
# where that depth came from. See pose3d.py for the camera model.

int32   track_id            # persistent across frames; survives brief occlusion
string  shape               # circle | triangle | rectangle | pentagon | trapezoid
float32 confidence          # tracker's label vote: fraction agreeing, 0..1
string  state               # confirmed | coasting | new

geometry_msgs/Point center_px   # image coords: x=u, y=v, z=0

# Position in the camera optical frame, in METRES (REP-103). The pipeline works
# internally in inches; the conversion happens at the publish boundary.
geometry_msgs/Point position
bool                has_position    # false until a circle has been seen at least once

float32 area_px

# The refined contour, in image coordinates (x=u, y=v, z=0). This is the outline
# the centre and the area were measured from, not a polygon approximation of it.
geometry_msgs/Point[] outline
