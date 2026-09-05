"""
Background-agnostic shape detection.

The predecessor (detect_shapes.py) rests on two assumptions that hold for grass
and flat-filled shapes and break elsewhere:

  1. the background is textured and the shapes are perfectly flat
  2. each shape is a single uniform colour

"PennAir 2024 App Dynamic Hard.mp4" breaks both: the ground is dark asphalt and
the shapes are gradient-filled (purple->green, navy->yellow). Under (1) a steep
gradient reads as texture, so parts of a shape are lost; under (2) the colour
model splits one gradient shape into pieces.

This module removes both assumptions and never models the background at all:

  * FIGURE/GROUND by *high-frequency residual*. Texture is fine-scale variation;
    a gradient is coarse-scale. Subtract a blurred copy and the gradient cancels
    while the texture survives, so "smooth" stops meaning "flat" and starts
    meaning "free of fine detail" -- true of flat fills and gradients alike.
  * EDGE CLOSURE as a second, independent route, for backgrounds that are
    themselves smooth (asphalt, water, sky) where no texture cue exists.
  * VERIFICATION by boundary contrast, so a candidate from either route has to
    prove it is bounded by a real edge.
  * REFINEMENT and SPLITTING by watershed on the image gradient, which needs no
    fill-colour model.

Usage:  python detect_shapes_agnostic.py [image] [-o out.png] [--debug]
"""

import argparse
import cv2
import numpy as np


# --------------------------------------------------------------------------
# scale-adaptive parameters
# --------------------------------------------------------------------------

def auto_params(frame_shape):
    """Window sizes, scaled from the 960x540 reference the constants were tuned on."""
    scale = frame_shape[1] / 960.0
    odd = lambda v: max(3, int(round(v)) | 1)
    return {"win": odd(11 * scale),          # texture-measuring window
            "ksize": odd(9 * scale),         # mask clean-up kernel
            "rksize": odd(5 * scale),        # refinement clean-up kernel
            "pad": int(round(25 * scale)),
            "min_area": int(round(500 * scale * scale))}


# --------------------------------------------------------------------------
# cue 1 -- fine-scale texture
# --------------------------------------------------------------------------

def texture_energy(bgr, hp_sigma=4.0, win=None):
    """Local strength of *fine* detail, per pixel.

    The predecessor measured local standard deviation directly, which answers
    "is this flat?". That is the wrong question once a shape can carry a
    gradient: a smooth ramp has a large standard deviation while containing no
    detail at all, so it reads as texture and the shape is lost.

    Subtracting a blurred copy first fixes that. A gradient survives the blur
    almost unchanged, so it cancels in the difference; texture does not survive
    it, so texture is all that remains. Taking the local deviation *of that
    residual* therefore asks "is this free of fine detail?" -- which is true of
    a flat fill and a gradient fill equally, and false of grass and asphalt
    equally.

    Measured on the hard video: the worst shape interior falls from 8.5 to 2.5
    while the background holds at 11.0, widening the separation from 2.1x to
    4.5x.

    Computed across all three channels so that two colours of equal brightness
    still register as a boundary.
    """
    win = auto_params(bgr.shape)["win"] if win is None else win
    img = bgr.astype(np.float32)

    # Low-pass by two box blurs rather than a Gaussian. Repeated box blur
    # converges on a Gaussian, and each pass is O(1) per pixel regardless of
    # radius, which a true Gaussian of this width is not.
    r = max(int(round(hp_sigma * 2)) | 1, 3)
    low = cv2.blur(cv2.blur(img, (r, r)), (r, r))
    residual = img - low

    # Collapse the channels before measuring, not after: one variance pass over
    # the colour-distance of the residual answers the same question as three
    # separate passes, and an edge between two equally bright colours still
    # registers because the chroma channels carry it.
    energy_in = cv2.sqrt((residual * residual).sum(axis=2))
    mean = cv2.boxFilter(energy_in, -1, (win, win))
    mean_sq = cv2.boxFilter(energy_in * energy_in, -1, (win, win))
    return cv2.sqrt(cv2.max(mean_sq - mean * mean, 0))


# --------------------------------------------------------------------------
# cue 2 -- closed boundaries
# --------------------------------------------------------------------------

def boundary_energy(bgr, blur=5):
    """Gradient magnitude after edge-preserving smoothing.

    A shape boundary is a step; texture is noise. A median filter flattens the
    noise while leaving the step intact, so what is left in the gradient is
    dominated by real boundaries. Taken per channel and combined, so an edge
    between two equally bright colours is not missed.
    """
    sm = cv2.medianBlur(bgr, blur)
    mag = np.zeros(bgr.shape[:2], np.float32)
    for c in range(3):
        ch = sm[:, :, c].astype(np.float32)
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.maximum(mag, cv2.magnitude(gx, gy))
    return mag


def relative_threshold(energy, frac=0.35, lo=1.0, hi=60.0):
    """Cut scaled to the image's own energy, so no absolute level is assumed.

    This is what makes the detector indifferent to the background: bright grass,
    dark asphalt and a smooth wall all produce different absolute energies, and
    the threshold follows whatever this frame happens to contain.
    """
    return float(np.clip(frac * np.median(energy), lo, hi))


# --------------------------------------------------------------------------
# candidate regions
# --------------------------------------------------------------------------

def _components(mask, min_area, max_area):
    """Outlines of a mask's connected regions, within a plausible size band.

    Labelled components rather than RETR_EXTERNAL contours, because the regions
    of interest are not always the outermost ones. On a smooth background the
    non-edge area is one sheet with the shapes punched out of it as separate
    components -- geometrically *inside* that sheet's outer contour, so
    RETR_EXTERNAL never returns them and every shape is silently dropped.
    Labelling sees them; nesting does not matter to it.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (min_area <= area <= max_area):
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        piece = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            out.append(max(cnts, key=cv2.contourArea) + (x, y))
    return out


def candidates(bgr, min_area, work_width=800, max_frac=0.40):
    """Regions that might be shapes, gathered from two independent routes.

    Route A (texture): interiors free of fine detail. Carries the load whenever
    the background has any texture at all -- grass, asphalt, gravel, carpet.

    Route B (enclosure): areas sealed off by a strong closed boundary. This is
    the answer for a background with *no* texture to contrast against -- a
    painted wall, still water, sky -- where route A sees the entire frame as
    smooth and abstains.

    The routes fail in opposite circumstances, so between them they cover
    backgrounds that neither covers alone, and each is thresholded relative to
    the frame's own statistics rather than to any absolute level. Candidates are
    collected per route and then merged, because unioning the raw masks would
    weld a smooth background to the shapes sitting on it and yield one blob.

    Stage 1 only has to *locate* shapes -- the watershed supplies the precision
    -- so it runs downscaled. 800 px wide is where accuracy stops improving;
    at 720 the grass trapezoid's area error trebles.

    `max_frac` is what lets a route abstain: on a smooth background route A
    marks nearly everything quiet, and a region covering 40% of the frame is
    rejected as a description of the background rather than of a shape.
    """
    full_h, full_w = bgr.shape[:2]
    scale = work_width / full_w if full_w > work_width else 1.0
    small = (cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
             if scale < 1.0 else bgr)
    p = auto_params(small.shape)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p["ksize"],) * 2)
    sh, sw = small.shape[:2]
    lo = max(int(min_area * scale * scale), 20)
    hi = max_frac * sh * sw

    # --- route A: quiet interiors -----------------------------------------
    energy = texture_energy(small, win=p["win"])
    quiet = (energy < relative_threshold(energy)).astype(np.uint8) * 255
    quiet = cv2.morphologyEx(quiet, cv2.MORPH_CLOSE, k, iterations=2)
    quiet = cv2.morphologyEx(quiet, cv2.MORPH_OPEN, k, iterations=3)
    found = _components(quiet, lo, hi)

    # --- route B: regions sealed off by an edge ----------------------------
    # Take the *complement* of the edges. A region the edges enclose becomes its
    # own component; the open background stays one large component and is
    # rejected by `hi`. Reading the enclosure this way, rather than filling the
    # edge map's outer contour, is what stops one connected web of texture
    # edges from swallowing the frame.
    mag_small = boundary_energy(small)
    mag = mag_small
    edges = (mag > max(np.percentile(mag, 92), 8.0)).astype(np.uint8) * 255
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k)
    interior = cv2.bitwise_not(edges)
    interior = cv2.morphologyEx(interior, cv2.MORPH_OPEN, k)
    found += _components(interior, lo, hi)

    # --- merge -------------------------------------------------------------
    # The routes agree often, so drop the duplicates, preferring the larger
    # outline since route A's is inset by half a texture window.
    found.sort(key=cv2.contourArea, reverse=True)
    kept, claimed = [], np.zeros((sh, sw), np.uint8)
    for c in found:
        probe = np.zeros((sh, sw), np.uint8)
        cv2.drawContours(probe, [c], -1, 255, cv2.FILLED)
        overlap = cv2.countNonZero(cv2.bitwise_and(probe, claimed))
        if overlap > 0.5 * cv2.countNonZero(probe):
            continue
        kept.append(c)
        claimed = cv2.bitwise_or(claimed, probe)

    if scale < 1.0:
        inv = 1.0 / scale
        kept = [np.round(c.astype(np.float32) * inv).astype(np.int32) for c in kept]
        energy = cv2.resize(energy, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
        mag = cv2.resize(mag, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
    return kept, energy, mag


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

def _crop_around(img, contour, margin):
    """Window an image to one contour's neighbourhood, with the contour re-based.

    Every per-shape measurement needs a mask the size of what it measures, not
    the size of the frame. Allocating and dilating full 1080p masks per shape
    cost more than the whole rest of the pipeline.
    """
    x, y, w, h = cv2.boundingRect(contour)
    x0, y0 = max(x - margin, 0), max(y - margin, 0)
    x1, y1 = min(x + w + margin, img.shape[1]), min(y + h + margin, img.shape[0])
    return img[y0:y1, x0:x1], contour - (x0, y0), (x0, y0)


def _median(values, cap=4000):
    """Median of at most `cap` samples -- enough for a ratio, far cheaper."""
    if values.size == 0:
        return 0.0
    if values.size > cap:
        values = values[:: values.size // cap]
    return float(np.median(values))


def verify(mag, energy, contour):
    """Score a candidate two ways: on its edge, and on its smoothness.

    Both scores describe the same rim, so they share one crop, one filled mask
    and one distance transform. Measured separately this was the single most
    expensive thing in the detector -- four 21x21 dilations per shape.

    Returns (boundary_contrast, texture_contrast, ring_energy):

      * boundary contrast -- gradient energy on the outline versus just outside
        it. A real shape sits on a real edge.
      * texture contrast  -- fine-detail energy outside versus inside. A real
        shape is smoother than the ground it lies on.

    They fail on different shapes, which is why both are computed. The olive
    trapezoid on green grass scores only 1.20 on the first (two greens meeting
    make almost no edge) but 47 on the second. A spurious patch of unusually
    even grass scores 1.07 and 1.08 -- it passes neither. Accepting a candidate
    on either score keeps the trapezoid and still rejects the patch.

    On a smooth background nothing is rough, so texture contrast goes quiet and
    boundary contrast decides; on a low-contrast boundary it is the other way
    round. The pair covers what neither covers alone.
    """
    sub_mag, local, _ = _crop_around(mag, contour, margin=60)
    sub_en, _, _ = _crop_around(energy, contour, margin=60)

    filled = np.zeros(sub_mag.shape, np.uint8)
    cv2.drawContours(filled, [local], -1, 255, cv2.FILLED)
    rim = np.zeros(sub_mag.shape, np.uint8)
    cv2.drawContours(rim, [local], -1, 255, 3)

    # One distance transform replaces every dilation: the ring is simply the
    # band between two distances from the shape, and the interior is the band
    # inside it. Cost is independent of how wide those bands are.
    outside = cv2.distanceTransform(255 - filled, cv2.DIST_L2, 3)
    inside = cv2.distanceTransform(filled, cv2.DIST_L2, 3)
    ring = (outside > 10) & (outside <= 30)
    # The interior band has to scale with the shape. `energy` is measured through
    # a window ~21 px wide, so a fixed 4 px inset samples a band that is still
    # half boundary -- harmless on a shape 200 px across, decisive on one 60 px
    # across, where that contaminated band is most of what gets measured and the
    # interior reads as rough as the ground. Taking the innermost quarter instead
    # keeps the sample clear of the outline at every size.
    core = inside > max(4.0, 0.25 * float(inside.max()))

    on = sub_mag[rim > 0]
    around = sub_mag[ring]
    bc = 0.0 if on.size == 0 or around.size == 0 else \
        _median(on) / (_median(around) + 1e-6)

    ring_energy = _median(sub_en[ring]) if ring.any() else 0.0
    tc = 0.0 if not core.any() or not ring.any() else \
        ring_energy / (_median(sub_en[core]) + 1e-6)

    # How rough the ground around this candidate actually is. Texture contrast
    # only means something where there is texture to contrast against, so this
    # is what decides which of the two scores is worth listening to.
    return bc, tc, ring_energy


# --------------------------------------------------------------------------
# refinement -- no fill-colour model
# --------------------------------------------------------------------------

def _watershed_once(bgr, seed_contour, win, r_out):
    """One flood, with `r_out` pixels of clearance before certain background."""
    h, w = bgr.shape[:2]
    pad = r_out + win
    x, y, bw, bh = cv2.boundingRect(seed_contour)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + bw + pad, w), min(y + bh + pad, h)
    roi = np.ascontiguousarray(bgr[y0:y1, x0:x1])

    seed = np.zeros(roi.shape[:2], np.uint8)
    cv2.drawContours(seed, [seed_contour - (x0, y0)], -1, 255, cv2.FILLED)

    r_in = max(win // 2, 3)
    sure_fg = cv2.erode(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r_in,) * 2))
    if cv2.countNonZero(sure_fg) == 0:
        return None

    # "Further than r_out from the seed" as a distance transform rather than a
    # dilation: a 139-pixel structuring element is enormously expensive, while
    # the distance transform costs the same whatever the radius.
    away = cv2.distanceTransform(255 - seed, cv2.DIST_L2, 3)

    markers = np.zeros(roi.shape[:2], np.int32)
    markers[away > r_out] = 1            # certainly background
    markers[sure_fg > 0] = 2             # certainly shape
    cv2.watershed(roi, markers)          # the image decides everything between

    out = (markers == 2).astype(np.uint8) * 255

    # A watershed line follows pixel boundaries, so the outline arrives with a
    # stair-step ripple that approxPolyDP reads as extra corners -- a triangle
    # came back a hexagon. Smoothing well below the shape scale but above the
    # ripple settles the vertex count without moving the edge.
    rk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (auto_params(bgr.shape)["rksize"],) * 2)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, rk)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, rk)

    cnts, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea) + (x0, y0)


def refine_watershed(bgr, seed_contour, win, solid_enough=0.95):
    """Snap a rough seed onto the true boundary using watershed.

    The predecessor sharpened outlines by sampling the fill colour and
    re-thresholding on colour distance. That needs the shape to *have* a fill
    colour; a purple-to-green rectangle does not, and thresholding around its
    mean keeps only the middle band.

    Watershed asks nothing about colour. It floods outward from what is known to
    be inside and inward from what is known to be outside, and the two meet on
    the strongest ridge between them -- the real edge, whatever colours lie
    either side of it.

    How much clearance to leave before "certainly background" is the one real
    parameter, and it cannot be fixed in advance:

      * too little and a sharp corner pokes outside the cleared band, is
        labelled certain background, and gets flooded away -- a triangle came
        back as a hexagon 13% too small;
      * too much and a *weak* edge loses the competition to some stronger ridge
        further out. The olive trapezoid on green grass leaks into the lawn and
        overshoots its true area by 37%.

    Convexity settles it. Every target shape is convex, so a correct outline is
    solid and a leak is visibly concave -- the leaking trapezoid falls to 0.72
    solidity while every correct outline holds above 0.96. So take the widest
    clearance that still yields a convex result, and only narrow when it doesn't.
    In the common case the first attempt is accepted and this costs one flood.
    The ladder starts wide and narrows; the widest rung that stays convex wins.
    """
    best, best_solidity = None, -1.0
    for mult in (4, 3, 2, 1):
        c = _watershed_once(bgr, seed_contour, win, max(win * mult, 5))
        if c is None:
            continue
        area = cv2.contourArea(c)
        solidity = area / max(cv2.contourArea(cv2.convexHull(c)), 1.0)
        if solidity > best_solidity:
            best, best_solidity = c, solidity
        if solidity >= solid_enough:
            return c
    return best if best is not None else seed_contour


# --------------------------------------------------------------------------
# splitting overlapping shapes -- no fill-colour model
# --------------------------------------------------------------------------

def split_merged(bgr, contour, min_area, peak_frac=0.55):
    """Separate shapes that overlapped into one region.

    The predecessor split a merged blob by clustering its fill colours, which
    needs each shape to *have* one. Two gradient-filled shapes defeat that
    completely -- a single purple-to-green rectangle contains more colour
    variation than the gap between two different shapes.

    Distance geometry replaces colour. The distance transform gives every pixel
    its distance to the nearest edge, so a convex shape has exactly one broad
    maximum at its middle. Two shapes overlapping produce a waisted region with
    *two* separated maxima. Seeding a watershed from those maxima cuts the union
    at its narrowest point, which is where the shapes actually meet.

    Counting maxima is itself the test for whether a split is warranted: a lone
    shape offers only one, so nothing happens.
    """
    x, y, w, h = cv2.boundingRect(contour)
    pad = 5
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, bgr.shape[1]), min(y + h + pad, bgr.shape[0])
    roi = np.ascontiguousarray(bgr[y0:y1, x0:x1])

    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour - (x0, y0)], -1, 255, cv2.FILLED)

    # Two convex shapes can only merge into a concave union, so a solid region
    # is a single shape and needs no examination.
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area <= 0 or cv2.contourArea(contour) / hull_area > 0.95:
        return [contour]

    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dt.max() <= 0:
        return [contour]
    peaks = (dt > peak_frac * dt.max()).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(peaks, 8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 20]
    if len(keep) < 2:
        return [contour]

    markers = np.zeros(roi.shape[:2], np.int32)
    markers[cv2.dilate(mask, np.ones((5, 5), np.uint8)) == 0] = 1
    for j, i in enumerate(keep, start=2):
        markers[labels == i] = j
    cv2.watershed(roi, markers)

    out = []
    for j in range(2, len(keep) + 2):
        piece = ((markers == j) & (mask > 0)).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) >= min_area:
                out.append(c + (x0, y0))
    return out if len(out) >= 2 else [contour]


def appearance(bgr, contour, bins=(12, 12)):
    """A hue/saturation histogram of a shape's interior, as its identity.

    The predecessor identified a shape across frames by its mean fill colour.
    That is exactly what a gradient destroys: the mean of a purple-to-green
    rectangle names a colour the shape does not contain, and it shifts as parts
    of the shape leave the frame or fall behind something else.

    A histogram has no such problem. It records *which* colours are present
    rather than averaging them away, so a two-tone shape is described by its two
    tones, and a partly hidden one still matches -- the bars shrink, but the
    occupied bins do not move. Hue and saturation, with value dropped, so that a
    shadow crossing the shape does not rename it.
    """
    x, y, w, h = cv2.boundingRect(contour)
    x0, y0 = max(x, 0), max(y, 0)
    roi = bgr[y0:y0 + h, x0:x0 + w]
    if roi.size == 0:
        return None
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.drawContours(mask, [contour - (x0, y0)], -1, 255, cv2.FILLED)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))     # keep clear of the rim
    if cv2.countNonZero(mask) < 20:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mask, list(bins), [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------

def classify(contour):
    """Name the shape from its polygon approximation and circularity.

    Geometric throughout, so it is already background- and fill-agnostic. It
    differs from the predecessor's in one place: how a circle is recognised.

    That test used circularity > 0.90, or a scattered vertex count (consensus
    below 0.7) with circularity > 0.80. The hard video's circle misses both by a
    hair -- circularity 0.899, consensus 0.72 -- and came back an "8-gon".
    Rather than nudge either threshold, note what the vertex vote actually said:
    it settled on **eight**. No real polygon here ever wins with more than six;
    the measured circles win with eight every time. A high winning vertex count
    is itself circle evidence, and it fails differently from circularity.
    """
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (peri * peri + 1e-9)

    counts = {}
    for eps in np.arange(0.010, 0.055, 0.0025):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        counts.setdefault(len(approx), []).append(approx)
    total = sum(len(v) for v in counts.values())
    n, approxes = max(counts.items(), key=lambda kv: (len(kv[1]), -kv[0]))
    approx = approxes[len(approxes) // 2]
    consensus = len(approxes) / total

    if circularity > 0.90 or (circularity > 0.80 and (consensus < 0.7 or n >= 7)):
        return "circle", approx

    if n == 3:
        return "triangle", approx
    if n == 4:
        p = approx.reshape(4, 2).astype(np.float32)
        sides = [np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)]
        skew = max(abs(sides[0] - sides[2]) / max(sides[0], sides[2]),
                   abs(sides[1] - sides[3]) / max(sides[1], sides[3]))
        return ("rectangle" if skew < 0.1 else "trapezoid"), approx
    return ({5: "pentagon", 6: "hexagon"}.get(n, f"{n}-gon")), approx


def _drop_repeating(results, group_min=8, area_tol=0.15):
    """Discard candidates that are one cell of a repeating pattern.

    A patterned background -- tiles, checkerboard, brickwork -- is made of
    regions that are individually indistinguishable from shapes: uniform inside,
    bounded by a strong edge. Every per-region test passes, because each cell
    really does look like a shape.

    What gives them away is that there are so many of them, all alike. A scene's
    actual shapes differ in size and kind; a tiling repeats one cell dozens of
    times. So a large group of candidates sharing a class and a size is read as
    background pattern rather than as subject matter. The threshold is set well
    above any plausible number of real shapes, and the rule is deliberately
    conservative: it costs a scene that genuinely holds eight-plus identical
    shapes, and buys immunity to a tiled floor.
    """
    if len(results) < group_min:
        return results
    drop = set()
    for i, a in enumerate(results):
        group = [j for j, b in enumerate(results)
                 if b["shape"] == a["shape"]
                 and abs(b["area"] - a["area"]) <= area_tol * max(a["area"], 1)]
        if len(group) >= group_min:
            drop.update(group)
    return [r for i, r in enumerate(results) if i not in drop]


def detect(bgr, min_area=None, min_contrast=4.0, min_tex_contrast=2.0,
           texture_floor=3.0):
    """Detect every shape in one frame. Pure function of that frame alone.

    Nothing here models the background: the thresholds are relative to the
    frame's own statistics, the refinement follows image gradients rather than
    colours, and candidates must earn their place by boundary contrast.
    """
    p = auto_params(bgr.shape)
    min_area = p["min_area"] if min_area is None else min_area

    # Both verifier maps come back from the candidate stage, computed once at its
    # working resolution. They feed ratios of medians, which downscaling leaves
    # intact, so there is nothing to gain from recomputing them at full size.
    seeds, energy, mag = candidates(bgr, min_area)
    h, w = bgr.shape[:2]

    results = []
    for seed in seeds:
        refined = refine_watershed(bgr, seed, p["win"])
        for c in split_merged(bgr, refined, min_area):
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            # Anything can look quiet or look enclosed by accident, so a candidate
            # has to earn its place. Which evidence to demand depends on what the
            # ground around it is actually like, and only one of the two tests is
            # informative in any given situation:
            #
            #   textured ground -> smoothness is decisive. Measured over both
            #     videos this alone admits every real shape and *zero* false
            #     positives, where the edge test lets ragged patches of asphalt
            #     speckle through.
            #   smooth ground   -> nothing is rough, so texture contrast is
            #     meaningless for every candidate alike and the edge is the only
            #     thing that distinguishes a shape from its surroundings.
            #
            # Accepting on either score unconditionally would import the weaker
            # test's false positives into the case where the stronger one works.
            contrast, tex, ring_energy = verify(mag, energy, c)
            if ring_energy >= texture_floor:
                if tex < min_tex_contrast:
                    continue
            elif contrast < min_contrast:
                continue

            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx, cy = int(round(m["m10"] / m["m00"])), int(round(m["m01"] / m["m00"]))

            bx, by, bw, bh = cv2.boundingRect(c)
            edge = 2
            partial = bx <= edge or by <= edge or bx + bw >= w - edge or by + bh >= h - edge

            hull = cv2.convexHull(c)
            solidity = area / max(cv2.contourArea(hull), 1.0)
            occluded = solidity < 0.97 and not partial
            if occluded:
                mh = cv2.moments(hull)
                if mh["m00"]:
                    cx = int(round(mh["m10"] / mh["m00"]))
                    cy = int(round(mh["m01"] / mh["m00"]))

            name, approx = classify(c)
            results.append({"contour": c, "approx": approx, "shape": name,
                            "center": (cx, cy), "area": area, "partial": partial,
                            "solidity": solidity, "occluded": occluded,
                            "contrast": contrast, "tex_contrast": tex,
                            "ring_energy": ring_energy,
                            "appearance": appearance(bgr, c)})

    results = _drop_repeating(results)
    results.sort(key=lambda r: -r["area"])
    return results, energy, mag


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def annotate(bgr, results):
    """Draw each shape's outline, centre and label."""
    out = bgr.copy()
    h, w = out.shape[:2]
    s = max(w / 960.0, 1.0)
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(round(2 * s)))
    for r in results:
        cx, cy = r["center"]
        cv2.drawContours(out, [r["contour"]], -1, (0, 255, 255), max(2, int(round(2 * s))))
        cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS,
                       int(round(20 * s)), thick)
        cv2.circle(out, (cx, cy), max(3, int(round(4 * s))), (0, 0, 255), -1)

        label = f'{r["shape"]} ({cx}, {cy})'
        fs = 0.55 * s
        (tw, th), _ = cv2.getTextSize(label, font, fs, thick)
        bx, by, bw, bh = cv2.boundingRect(r["contour"])
        ty = by - int(8 * s) if by - th - int(12 * s) >= 0 else by + bh + th + int(10 * s)
        tx = int(np.clip(bx + bw // 2 - tw // 2, 2, w - tw - 2))
        ty = int(np.clip(ty, th + int(6 * s), h - 4))
        cv2.rectangle(out, (tx - 4, ty - th - 5), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(out, label, (tx, ty), font, fs, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default="PennAir 2024 App Static.png")
    ap.add_argument("-o", "--output", default="output_static_agnostic.png")
    ap.add_argument("--debug", action="store_true",
                    help="also write the texture map and the candidate mask")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"could not read {args.image}")

    results, energy, mag = detect(bgr)
    cv2.imwrite(args.output, annotate(bgr, results))

    print(f"{len(results)} shapes detected in {args.image}")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['shape']:<10} center=({r['center'][0]:4d}, {r['center'][1]:4d})"
              f"  area={r['area']:.0f} px"
              f"  edge={r['contrast']:.1f}  smoothness={r['tex_contrast']:.1f}")
    print(f"wrote {args.output}")

    if args.debug:
        cv2.imwrite("debug_texture_agnostic.png",
                    cv2.normalize(energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        cv2.imwrite("debug_edges_agnostic.png",
                    cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        print("wrote debug_texture_agnostic.png, debug_edges_agnostic.png")


if __name__ == "__main__":
    main()
