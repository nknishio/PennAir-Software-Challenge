"""
Detect solid shapes on a grassy background and mark their centers.

Approach
--------
Colour thresholding is a trap here: one of the shapes is bright green, the same
hue as the grass. What actually separates figure from ground is *texture* --
grass is high-frequency noise, the shapes are perfectly flat fills. So we
segment on local standard deviation of intensity: smooth pixels are shapes.

    std(x,y) = sqrt( E[I^2] - E[I]^2 )   over a local window

Usage:  python detect_shapes.py [image] [-o out.png] [--debug]
"""

import argparse
import cv2
import numpy as np


def smoothness_mask(bgr, win=11, blur=5):
    """Binary mask of locally-flat (low texture) regions."""
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
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=3)

    std_u8 = np.clip(std, 0, 40) * (255.0 / 40.0)
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


def refine_by_colour(bgr, seed_contour, tol=40, pad=25):
    """Recover a shape's exact outline from a texture seed.

    The variance window blurs across each edge, so the seed is inset and its
    corners are rounded by the morphology. But each shape is a single flat
    colour: sample that colour from the seed's interior and re-threshold the
    surrounding patch on colour distance, which snaps to the true edge.
    """
    h, w = bgr.shape[:2]
    x, y, bw, bh = cv2.boundingRect(seed_contour)
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + bw + pad, w), min(y + bh + pad, h)
    roi = bgr[y0:y1, x0:x1]

    seed = np.zeros(roi.shape[:2], np.uint8)
    cv2.drawContours(seed, [seed_contour - (x0, y0)], -1, 255, cv2.FILLED)
    fill = cv2.mean(roi, mask=seed)[:3]

    dist = np.linalg.norm(roi.astype(np.float32) - np.float32(fill), axis=2)
    m = (dist < tol).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return seed_contour
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < cv2.contourArea(seed_contour):
        return seed_contour                     # refinement lost ground; keep seed
    return best + (x0, y0)


def detect(bgr, min_area=500):
    mask, std_u8 = smoothness_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    results = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        # Drop anything hugging the frame border (partial / background blobs).
        x, y, bw, bh = cv2.boundingRect(c)
        if bw >= 0.98 * w and bh >= 0.98 * h:
            continue

        c = refine_by_colour(bgr, c)
        area = cv2.contourArea(c)       # measure the refined outline, not the seed

        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx, cy = int(round(m["m10"] / m["m00"])), int(round(m["m01"] / m["m00"]))

        name, approx = classify(c)
        results.append({"contour": c, "approx": approx, "shape": name,
                        "center": (cx, cy), "area": area})

    results.sort(key=lambda r: (-r["area"]))
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
