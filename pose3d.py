"""
Camera model: pixel centres -> metric (X, Y, Z) in the camera frame.

Everything upstream of this file answers *where in the image* a shape is. A drone
needs to know *where in the world*, and the gap between the two is a camera model
plus one known length.

    K = [[2564.3186869,   0,            0],
         [   0,        2569.70273111,   0],
         [   0,           0,            1]]

    the circle has radius 10 in

Two assumptions carry the whole thing, both stated in the brief:

  1. **Flat surface, fronto-parallel.** The shapes are coplanar and that plane is
     parallel to the sensor, so one depth Z describes the entire scene. This is
     what makes a single known length enough for every shape at once.
  2. **The circle is the ruler.** It is the only object whose true size is given.

Depth from the circle
---------------------
A circle of radius R lying on a fronto-parallel plane at depth Z projects to an
ellipse with semi-axes (fx*R/Z, fy*R/Z) px, so its pixel area is

    A_px = pi * fx * fy * R^2 / Z^2        =>    Z = R * sqrt(pi * fx * fy / A_px)

Area is used rather than a measured radius on purpose: area integrates every
boundary pixel, where a radius is read off one or two of them. The same relation
written for an arbitrary metric area A_m generalises it to any shape:

    Z = sqrt(fx * fy * A_m / A_px)         and      A_m = A_px * Z^2 / (fx * fy)

Back-projection
---------------
    X = (u - cx) * Z / fx        Y = (v - cy) * Z / fy        Z = Z

**On the principal point.** The supplied K has cx = cy = 0, which places the
optical axis at the top-left pixel rather than the image centre. That is used
here exactly as given, so X and Y are measured from the top-left corner's line of
sight and are consequently large and single-signed. Passing ``--pp center``
recentres it on the image, which is what a calibrated camera would normally
report; the depth Z is identical either way, since Z depends only on fx and fy.

**On resolution.** fx of ~2564 px belongs to the 1920-wide footage. Intrinsics
are in pixels, so they scale with the image: a frame resized by s has its K
scaled by s too. `Camera.for_frame` does this, which is what lets the 960-wide
static PNG and the 1080p video be measured with one calibration.

Carrying the scale when the circle is gone
------------------------------------------
The circle is not visible in every frame. While it *is* visible, its depth also
reveals the true size of everything else on the plane -- one division per shape --
and a shape whose metric size is known is itself a ruler from then on. So the
scale is bootstrapped once from the circle and afterwards carried by whichever
shapes happen to be in view. `PlaneScale` holds that memory, keyed by track id,
and reads only past frames: nothing here breaks the streaming contract.
"""

import cv2
import numpy as np
from collections import deque

# --------------------------------------------------------------------------
# the given calibration
# --------------------------------------------------------------------------

K_GIVEN = np.array([[2564.3186869, 0.0, 0.0],
                    [0.0, 2569.70273111, 0.0],
                    [0.0, 0.0, 1.0]])

K_REF_WIDTH = 1920          # resolution the calibration belongs to
CIRCLE_RADIUS_IN = 10.0     # the one known length in the whole problem

IN_PER_M = 39.3700787
IN_PER_FT = 12.0
UNITS = {"in": 1.0, "ft": 1.0 / IN_PER_FT, "m": 1.0 / IN_PER_M}


class Camera:
    """A pinhole camera, resolution-aware."""

    def __init__(self, K=None, ref_width=K_REF_WIDTH, principal="given"):
        K = K_GIVEN if K is None else np.asarray(K, np.float64)
        self.K0 = K.copy()
        self.ref_width = float(ref_width)
        self.principal = principal
        self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
        self.cx, self.cy = float(K[0, 2]), float(K[1, 2])

    def for_frame(self, frame_shape):
        """This camera expressed in the pixels of a frame of this size.

        Intrinsics are measured in pixels, so resizing an image by s multiplies
        every entry of K except the homogeneous 1 by s. Skipping this is the
        classic way to get a depth that silently changes when you pass --scale.
        """
        h, w = frame_shape[:2]
        s = w / self.ref_width
        cam = Camera(self.K0, self.ref_width, self.principal)
        cam.fx, cam.fy = self.fx * s, self.fy * s
        if self.principal == "center":
            cam.cx, cam.cy = w / 2.0, h / 2.0
        else:
            cam.cx, cam.cy = self.cx * s, self.cy * s
        return cam

    @property
    def K(self):
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    # -- the two directions ------------------------------------------------

    def backproject(self, u, v, Z):
        """Pixel + depth -> metric (X, Y, Z) in the camera frame, inches."""
        return ((u - self.cx) * Z / self.fx,
                (v - self.cy) * Z / self.fy,
                Z)

    def project(self, X, Y, Z):
        """Metric point -> pixel. The inverse of backproject; used by the tests."""
        return (X * self.fx / Z + self.cx, Y * self.fy / Z + self.cy)

    # -- scale <-> depth ---------------------------------------------------

    def depth_from_area(self, area_px, metric_area):
        """Z = sqrt(fx*fy*A_m/A_px). Needs a fronto-parallel plane."""
        if area_px <= 0 or metric_area <= 0:
            return None
        return float(np.sqrt(self.fx * self.fy * metric_area / area_px))

    def metric_area(self, area_px, Z):
        """The inverse: how big this pixel area really is, given the depth."""
        return float(area_px * Z * Z / (self.fx * self.fy))

    def pixel_radius(self, metric_radius, Z):
        """Semi-axes a circle of this radius would have at this depth."""
        return (metric_radius * self.fx / Z, metric_radius * self.fy / Z)


CIRCLE_AREA_IN2 = float(np.pi * CIRCLE_RADIUS_IN ** 2)


# --------------------------------------------------------------------------
# the metric-size memory
# --------------------------------------------------------------------------

CIRCLE_SCORE_MIN = 0.88


def circle_score(contour):
    """How much of its own smallest enclosing circle this outline fills.

    The ruler has to be identified before it can be trusted, and the classifier's
    word alone is not enough: mistaking a pentagon for the circle does not lose
    the scale, it silently *falsifies* it -- a regular pentagon fills 0.757 of its
    circumcircle, so a pentagon that size read as the 10 in circle reports the scene
    ~15% *further away* than it is. A wrong depth is worse than no depth, so the
    label is corroborated by a measurement that fails differently.

    Circularity (4*pi*A/P^2) is the obvious candidate and the wrong one here: it
    is built on perimeter, which a ragged boundary inflates, so it is weakest
    exactly when segmentation is shaky. This ratio is area over area, and area is
    the most stable thing a contour has. Measured on the supplied footage the
    circle scores 0.947-0.957 and the next best shape 0.753 -- a margin wide
    enough that the threshold is not a tuned number.
    """
    (_, _), r = cv2.minEnclosingCircle(contour)
    if r <= 0:
        return 0.0
    return float(cv2.contourArea(contour) / (np.pi * r * r))


def is_measurable(det):
    """Only a whole, unobstructed outline may be used as a ruler.

    A clipped or occluded shape shows less area than it has, and area is the
    entire measurement here -- depth goes as 1/sqrt(area), so a shape showing
    half of itself reports being 41% further away than it is. This is the same
    guard the tracker puts on classification votes, for the same reason.
    """
    return not det.get("partial", False) and not det.get("occluded", False)


class PlaneScale:
    """Per-frame plane depth, bootstrapped from the circle and carried by the rest.

    Causal by construction: `update` sees the current frame's detections and its
    own memory of past frames, and nothing else.
    """

    def __init__(self, camera, memory=45, min_learn=5, max_rate=0.05,
                 max_jitter=0.05):
        self.cam = camera                # already scaled to the frame
        self.memory = memory             # frames of size history per shape
        self.min_learn = min_learn       # observations before a shape may rule
        self.max_rate = max_rate         # plausible depth change per frame
        self.max_jitter = max_jitter     # spread that disqualifies a ruler
        self.sizes = {}                  # key -> deque of metric-area estimates
        self.last_Z = None
        self.last_source = None

    # -- internals ---------------------------------------------------------

    def _history(self, key):
        """This shape's remembered size estimates, once there are enough to mean
        anything. Below `min_learn` observations a shape is not yet a ruler."""
        d = self.sizes.get(key)
        if d is None or len(d) < self.min_learn:
            return None
        return np.asarray(d, np.float64)

    def _known(self, key):
        a = self._history(key)
        return None if a is None else float(np.median(a))

    def _jitter(self, key):
        """How much this shape's measured size has wobbled, relative to itself.

        A ruler is only as good as the length marked on it. The circle's measured
        area barely moves; the trapezoid's -- whose boundary is the detector's
        known weak point -- swings by a quarter. Both are in the memory already,
        so the memory can say which one to believe: this is the interquartile
        spread of a shape's own size estimates, over its median.
        """
        a = self._history(key)
        if a is None:
            return None
        m = float(np.median(a))
        if m <= 0:
            return None
        return float(np.percentile(a, 75) - np.percentile(a, 25)) / m

    def _learn(self, key, area_px, Z):
        self.sizes.setdefault(key, deque(maxlen=self.memory)).append(
            self.cam.metric_area(area_px, Z))

    def _fuse(self, proxies):
        """One depth from several stand-in rulers, in two steps.

        Taking the median of everything on offer is the obvious move and it is
        not good enough: with only two rulers in view -- the common case once the
        circle has gone -- the median is their average, so one bad reading moves
        the answer by half its error. Measured on the hard footage, that is where
        every large depth error came from, and in each of those frames a *correct*
        estimate was sitting right beside the wrong one.

        So instead of averaging the disagreement away, pick between them, on two
        signals that fail differently:

          * a ruler whose own measured size has been steady is worth more than
            one that wobbles (`_jitter`);
          * and the plane does not teleport, so of the rulers that survive, the
            one nearest the last known depth is the one to believe.

        The second criterion could in principle lock in a drift, which is why it
        only ever chooses *between* independent measurements and never invents
        one -- and why the circle, whenever it is visible, overrides all of this.
        """
        steady = [z for z, j in proxies if j is not None and j <= self.max_jitter]
        pool = steady or [z for z, _ in proxies]
        if self.last_Z is not None:
            return min(pool, key=lambda z: abs(z - self.last_Z))
        return float(np.median(pool))

    # -- the one public step ----------------------------------------------

    def update(self, observations):
        """observations: [(key, shape_name, area_px, measurable, circle_score), ...]

        Returns (Z, source). source is 'circle' when the ruler itself was in
        view, 'learned' when the scale was carried by a shape the circle had
        previously measured, 'held' when neither was available and the last
        known depth is being reported, or None before any circle has been seen.
        """
        usable = [o for o in observations if o[3] and o[2] > 0]

        # 1. the ruler, if it is in view -- named a circle *and* shaped like one
        circ = [self.cam.depth_from_area(a, CIRCLE_AREA_IN2)
                for _, name, a, _, cs in usable
                if name == "circle" and cs >= CIRCLE_SCORE_MIN]
        circ = [z for z in circ if z]

        if circ:
            Z, source = float(np.median(circ)), "circle"
        else:
            # 2. anything the circle taught us
            proxies = []
            for key, _, area, _, _ in usable:
                A_m = self._known(key)
                if not A_m:
                    continue
                z = self.cam.depth_from_area(area, A_m)
                if z:
                    proxies.append((z, self._jitter(key)))

            if proxies:
                Z, source = self._fuse(proxies), "learned"
            elif self.last_Z is not None:
                Z, source = self.last_Z, "held"      # coast, and say so
            else:
                self.last_source = None
                return None, None                    # no scale exists yet

        # 3. one plane, so this depth also sizes everything else in view
        if source in ("circle", "learned"):
            for key, name, area, _, _ in usable:
                if name != "circle":
                    self._learn(key, area, Z)

        # A camera cannot jump. Depth is measured from area, and area is exactly
        # what a momentary bad segmentation gets wrong -- a trapezoid whose
        # boundary frays for one frame reads as much further away than it is. The
        # measured tail was 1.3% of frames more than 5% out, all of them single
        # frames, all of them carried by a learned ruler rather than the circle.
        #
        # What rules them out is not a better area but physics: at 30 fps, 5% of
        # 250 in is a climb rate of over 100 in/s, far beyond anything a drone
        # does. So the estimate is rate-limited against the previous frame, which
        # leaves a genuine altitude change free to track (it simply takes a few
        # frames to catch up, and test_pose3d checks that it does) while a
        # one-frame excursion is clipped to nothing. Causal, like the tracker.
        if self.last_Z is not None and self.max_rate:
            lo, hi = self.last_Z * (1 - self.max_rate), self.last_Z * (1 + self.max_rate)
            Z = float(np.clip(Z, lo, hi))

        self.last_Z, self.last_source = Z, source
        return Z, source

    def locate(self, center, Z):
        """Pixel centre + the plane depth -> (X, Y, Z) inches."""
        return self.cam.backproject(float(center[0]), float(center[1]), Z)


def convert(xyz, unit="in"):
    k = UNITS[unit]
    return tuple(v * k for v in xyz)


def solve_frame(detections, camera, scale=None, keys=None):
    """Attach 3D coordinates to one frame's detections. Static-image entry point.

    Each detection gains 'xyz' (inches, camera frame) and the frame's depth is
    returned alongside. `scale` may be an existing PlaneScale to carry memory
    across frames; without one, this frame must contain the circle.
    """
    scale = PlaneScale(camera) if scale is None else scale
    if keys is None:
        keys = list(range(len(detections)))
    obs = [(k, d["shape"], d["area"], is_measurable(d), circle_score(d["contour"]))
           for k, d in zip(keys, detections)]
    Z, source = scale.update(obs)
    for d in detections:
        d["xyz"] = scale.locate(d["center"], Z) if Z else None
        d["depth_source"] = source
    return Z, source, scale
