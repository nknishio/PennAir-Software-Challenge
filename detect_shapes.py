"""
Detect solid shapes on a grassy background and mark their centers.

Approach
--------
color thresholding is a trap here: one of the shapes is bright green, the same
hue as the grass. What actually separates figure from ground is *texture* --
grass is high-frequency noise, the shapes are perfectly flat fills. So we
segment on local standard deviation of intensity: smooth pixels are shapes.

    std(x,y) = sqrt( E[I^2] - E[I]^2 )   over a local window

Usage:  python detect_shapes.py [image] [-o out.png] [--debug]
"""

import argparse
import cv2
import numpy as np


def auto_params(frame_shape):
    """Scale the window sizes with resolution.

    Every constant here was tuned on a 960x540 image. Grass blades in a 1080p
    frame are twice as wide in pixels, so the windows that measure them have to
    grow too, otherwise the variance window can fit *inside* a single blade and
    read it as flat. Scaling by frame width keeps the algorithm resolution
    independent; at 960 wide it reproduces the original constants exactly.
    """
    scale = frame_shape[1] / 960.0
    odd = lambda v: max(3, int(round(v)) | 1)          # kernels must be odd
    return {"win": odd(11 * scale),
            "ksize": odd(9 * scale),
            "rksize": odd(5 * scale),
            "pad": int(round(25 * scale)),
            "min_area": int(round(500 * scale * scale))}   # area scales as length^2


def smoothness_mask(bgr, win=None, blur=5, ksize=None, work_width=960):
    """Binary mask of locally-flat (low texture) regions.

    Stage 1 only has to *locate* the shapes -- Stage 2 supplies the precision --
    so it runs on a downscaled copy. On 1080p that is 4x fewer pixels through the
    expensive morphology, and it puts the frame back at the width every constant
    here was tuned for. The threshold is relative to the median, so it
    self-calibrates to the lower contrast that downsampling produces.
    """
    full_h, full_w = bgr.shape[:2]
    scale = work_width / full_w if full_w > work_width else 1.0
    if scale < 1.0:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    p = auto_params(bgr.shape)
    win = p["win"] if win is None else win
    ksize = p["ksize"] if ksize is None else ksize

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (blur, blur), 0)      # kill sensor noise, keep grass texture

    mean = cv2.boxFilter(gray, -1, (win, win))
    mean_sq = cv2.boxFilter(gray * gray, -1, (win, win))
    std = cv2.sqrt(cv2.max(mean_sq - mean * mean, 0))

    # The flat shapes cover only a few percent of the frame, so Otsu would just
    # split the grass distribution in half. Instead scale the cut to the
    # background's own roughness: solid fills sit near zero, grass near median.
    thresh = float(np.clip(0.35 * np.median(std), 2.0, 20.0))
    mask = (std < thresh).astype(np.uint8) * 255

    # Close pinholes, then open to drop stray flat specks between grass blades.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=3)

    std_u8 = np.clip(std, 0, 40) * (255.0 / 40.0)
    if scale < 1.0:      # back to full resolution; Stage 2 sharpens the edges anyway
        mask = cv2.resize(mask, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
        std_u8 = cv2.resize(std_u8, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
    return mask, std_u8.astype(np.uint8)


def classify(contour):
    """Name the shape from its polygon approximation and circularity."""
    peri = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * area / (peri * peri + 1e-9)   # 1.0 for a perfect circle

    # A single approxPolyDP tolerance is brittle: too tight and contour noise
    # invents vertices, too loose and real corners merge. Sweep the tolerance
    # instead and let the vertex counts vote.
    counts = {}
    for eps in np.arange(0.010, 0.055, 0.0025):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        counts.setdefault(len(approx), []).append(approx)
    total = sum(len(v) for v in counts.values())
    n, approxes = max(counts.items(), key=lambda kv: (len(kv[1]), -kv[0]))
    approx = approxes[len(approxes) // 2]
    consensus = len(approxes) / total

    # A true polygon holds one vertex count across the whole sweep; a circle has
    # no natural count, so its votes scatter. That disagreement is a sharper
    # circle test than circularity alone, whose margin over a regular pentagon
    # (0.865 ideal) or hexagon (0.907) is uncomfortably thin.
    if circularity > 0.90 or (consensus < 0.7 and circularity > 0.80):
        return "circle", approx

    if n == 3:
        return "triangle", approx
    if n == 4:
        # Tell a square/rect from a trapezoid by comparing opposite side pairs.
        p = approx.reshape(4, 2).astype(np.float32)
        sides = [np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)]
        skew = max(abs(sides[0] - sides[2]) / max(sides[0], sides[2]),
                   abs(sides[1] - sides[3]) / max(sides[1], sides[3]))
        return ("rectangle" if skew < 0.1 else "trapezoid"), approx
    return ({5: "pentagon", 6: "hexagon"}.get(n, f"{n}-gon")), approx


def dominant_colors(roi, seed_mask, max_k=3, min_sep=60.0, min_frac=0.05,
                    max_samples=4000):
    """The distinct fill colors inside one blob -- usually one, more if shapes overlap.

    When two shapes overlap they merge into a single smooth blob, and the seed
    contour describes both at once. But each shape is a different flat color, so
    the blob's pixels form well-separated clusters in BGR space. k-means finds
    them; two guards decide whether a split is real:

      * min_sep  -- the cluster centers must be genuinely different colors, not
                    two halves of one shape's noise. A real merge measures >250
                    apart; splitting a single shape measures <5.
      * min_frac -- every cluster must hold a worthwhile share of the blob, which
                    rejects a handful of anti-aliased edge pixels posing as a
                    third shape.

    Raising k only pays if *both* hold, so a lone shape returns its single color.
    """
    px = roi[seed_mask > 0].astype(np.float32)
    if len(px) < 10:
        return [cv2.mean(roi, mask=seed_mask)[:3]]

    # Clustering needs the colour distribution, not every pixel of it. A few
    # thousand samples locate the centres just as well for a fraction of the cost.
    if len(px) > max_samples:
        px = px[np.random.default_rng(0).choice(len(px), max_samples, replace=False)]

    # Most blobs are a single shape. If the spread is far too small to hold two
    # colours min_sep apart, say so without paying for k-means at all.
    if px.std(axis=0).max() * 4 < min_sep:
        return [tuple(map(float, px.mean(axis=0)))]

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    best = [px.mean(axis=0)]
    for k in range(2, max_k + 1):
        _, labels, centers = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.ravel(), minlength=k)
        sep = min(np.linalg.norm(centers[i] - centers[j])
                  for i in range(k) for j in range(i + 1, k))
        if sep < min_sep or counts.min() < min_frac * len(px):
            break                          # this k is not justified; keep the last
        best = list(centers)
    return [tuple(map(float, c)) for c in best]


def refine_blob(bgr, seed_contour, min_area, tol=40, pad=None):
    """Turn one texture seed into one exact contour per shape it contains.

    Two jobs at once. The seed is inset -- the variance window straddles each
    edge -- and its corners are rounded by the morphology, so the outline needs
    sharpening against the original pixels. And if the seed merged several
    overlapping shapes, it needs splitting. The flat fill colors answer both.

    Returns a (contour, fill_color) pair per shape. The colour is free here and
    is what lets a tracker keep hold of a shape's identity across frames.
    """
    h, w = bgr.shape[:2]
    pad = auto_params(bgr.shape)["pad"] if pad is None else pad
    x, y, bw, bh = cv2.boundingRect(seed_contour)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + bw + pad, w), min(y + bh + pad, h)
    roi = bgr[y0:y1, x0:x1]

    seed = np.zeros(roi.shape[:2], np.uint8)
    cv2.drawContours(seed, [seed_contour - (x0, y0)], -1, 255, cv2.FILLED)

    # A fill colour close to the background (the olive trapezoid sits only ~70
    # from grass, where the others sit 180-260) lets scattered grass pixels pass
    # the colour test, fraying the outline into false vertices. They arrive as
    # speckle and hair, so an opening removes them; the closing then reseals the
    # shape. Costs ~6% of area -- which was the fringe, not the shape.
    rk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (auto_params(bgr.shape)["rksize"],) * 2)
    out = []
    for fill in dominant_colors(roi, seed):
        d2 = ((roi.astype(np.float32) - np.float32(fill)) ** 2).sum(axis=2)
        m = (d2 < tol * tol).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, rk)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, rk)

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        keep = []
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            # Keep only what overlaps the seed, so a same-colored shape that
            # happens to sit in the same crop is not absorbed into this one.
            probe = np.zeros_like(seed)
            cv2.drawContours(probe, [c], -1, 255, cv2.FILLED)
            if cv2.countNonZero(cv2.bitwise_and(probe, seed)) > 0:
                keep.append(c)

        # An occluder lying across the middle of a shape cuts its visible area
        # into two pieces. They are one shape -- same seed, same fill colour --
        # so rejoin them rather than reporting two. The gap they span has to be
        # small relative to the pieces themselves; genuinely separate shapes of
        # the same colour would sit much further apart.
        while len(keep) > 1:
            merged = False
            for a in range(len(keep)):
                for b in range(a + 1, len(keep)):
                    span = max(cv2.boundingRect(np.vstack([keep[a], keep[b]]))[2:])
                    reach = max(max(cv2.boundingRect(keep[a])[2:]),
                                max(cv2.boundingRect(keep[b])[2:]))
                    if span <= 1.6 * reach:
                        keep[a] = cv2.convexHull(np.vstack([keep[a], keep[b]]))
                        keep.pop(b)
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break

        out.extend((c + (x0, y0), fill) for c in keep)

    if not out:
        # Refinement found nothing; trust the seed and measure its colour directly.
        return [(seed_contour, cv2.mean(roi, mask=seed)[:3])]
    return out


def detect(bgr, min_area=None):
    """Detect every shape in one frame. Pure function of that frame alone."""
    min_area = auto_params(bgr.shape)["min_area"] if min_area is None else min_area
    mask, std_u8 = smoothness_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    results = []
    for seed in contours:
        if cv2.contourArea(seed) < min_area:
            continue
        # Drop anything spanning the whole frame -- the signature of a failed
        # segmentation, which should read as zero shapes rather than one absurd one.
        x, y, bw, bh = cv2.boundingRect(seed)
        if bw >= 0.98 * w and bh >= 0.98 * h:
            continue

        for c, fill in refine_blob(bgr, seed, min_area):
            area = cv2.contourArea(c)      # measure the refined outline, not the seed
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx, cy = int(round(m["m10"] / m["m00"])), int(round(m["m01"] / m["m00"]))

            # A shape running off the edge is only partly visible, so its centroid
            # and vertex count describe the visible piece, not the whole shape.
            bx, by, bbw, bbh = cv2.boundingRect(c)
            edge = 2
            partial = bx <= edge or by <= edge or bx + bbw >= w - edge or by + bbh >= h - edge

            # Solidity -- area over convex-hull area -- is ~1 for these shapes,
            # which are all convex. A bite out of the outline means something is
            # in front of it.
            hull = cv2.convexHull(c)
            solidity = area / max(cv2.contourArea(hull), 1.0)
            occluded = solidity < 0.97 and not partial
            if occluded:
                # The visible centroid is dragged away from the hidden side. The
                # hull spans the bite and pulls it back -- valid because the shape
                # is convex. (It cannot restore a corner that is fully covered;
                # the tracker's motion model handles that case.)
                mh = cv2.moments(hull)
                if mh["m00"]:
                    cx, cy = int(round(mh["m10"] / mh["m00"])), int(round(mh["m01"] / mh["m00"]))

            name, approx = classify(c)
            results.append({"contour": c, "approx": approx, "shape": name,
                            "center": (cx, cy), "area": area, "partial": partial,
                            "solidity": solidity, "occluded": occluded,
                            "color": tuple(map(float, fill))})

    results.sort(key=lambda r: -r["area"])
    return results, mask, std_u8


def annotate(bgr, results):
    out = bgr.copy()
    h, w = out.shape[:2]
    for r in results:
        cx, cy = r["center"]
        cv2.drawContours(out, [r["contour"]], -1, (0, 255, 255), 2)
        cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

        # Park the label just outside the shape so it never hides the centre.
        label = f'{r["shape"]} ({cx}, {cy})'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bx, by, bw, bh = cv2.boundingRect(r["contour"])
        ty = by - 8 if by - th - 12 >= 0 else by + bh + th + 10
        tx = int(np.clip(bx + bw // 2 - tw // 2, 2, w - tw - 2))
        ty = int(np.clip(ty, th + 6, h - 4))

        cv2.rectangle(out, (tx - 4, ty - th - 5), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default="PennAir 2024 App Static.png")
    ap.add_argument("-o", "--output", default="output_static.png")
    ap.add_argument("--debug", action="store_true", help="also write mask/std images")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"could not read {args.image}")

    results, mask, std_u8 = detect(bgr)
    out = annotate(bgr, results)
    cv2.imwrite(args.output, out)

    print(f"{len(results)} shapes detected in {args.image}")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['shape']:<10} center=({r['center'][0]:4d}, {r['center'][1]:4d})"
              f"  area={r['area']:.0f} px")
    print(f"wrote {args.output}")

    if args.debug:
        cv2.imwrite("debug_texture.png", std_u8)
        cv2.imwrite("debug_mask.png", mask)
        print("wrote debug_texture.png, debug_mask.png")


if __name__ == "__main__":
    main()
