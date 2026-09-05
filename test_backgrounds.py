"""
Background-agnosticism test.

Renders the same five shapes over a range of synthetic backgrounds -- smooth and
textured, light and dark, plain and patterned -- and checks that every one is
found, centred and named. Because the scene is generated here, ground truth is
exact rather than inferred, and the smooth backgrounds exercise the path that no
supplied footage covers.

Usage:  python test_backgrounds.py [--save out.png]
"""

import argparse
import cv2
import numpy as np

import detect_shapes_agnostic as da

W, H = 960, 540
RNG = np.random.default_rng(7)

SHAPES = [
    ("circle",    (200, 150), 62),
    ("triangle",  (470, 150), 74),
    ("rectangle", (740, 150), 60),
    ("pentagon",  (300, 390), 70),
    ("trapezoid", (640, 390), 72),
]


def shape_points(kind, c, r):
    cx, cy = c
    if kind == "triangle":
        return np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
    if kind == "rectangle":
        return np.array([[cx - r, cy - r], [cx + r, cy - r],
                         [cx + r, cy + r], [cx - r, cy + r]], np.int32)
    if kind == "pentagon":
        a = np.linspace(-np.pi / 2, 3 * np.pi / 2, 6)[:5]
        return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1).astype(np.int32)
    if kind == "trapezoid":
        return np.array([[cx - r // 2, cy - r], [cx + r // 2, cy - r],
                         [cx + r, cy + r], [cx - r, cy + r]], np.int32)
    return None                      # circle drawn directly


def backgrounds():
    """A spread of grounds: smooth, textured, light, dark, patterned."""
    out = {}

    out["solid mid-grey"] = np.full((H, W, 3), 128, np.uint8)
    out["solid blue"] = np.full((H, W, 3), (170, 90, 40), np.uint8)
    out["solid white"] = np.full((H, W, 3), 245, np.uint8)

    g = np.linspace(30, 220, W, dtype=np.float32)
    out["smooth gradient"] = np.dstack([np.tile(g, (H, 1))] * 3).astype(np.uint8)

    fine = RNG.normal(140, 18, (H, W, 1)).clip(0, 255)
    out["fine noise (sand)"] = np.repeat(fine, 3, axis=2).astype(np.uint8)

    coarse = RNG.normal(70, 45, (H // 3, W // 3, 1)).clip(0, 255)
    coarse = cv2.resize(coarse, (W, H), interpolation=cv2.INTER_LINEAR)[:, :, None]
    out["coarse noise (gravel)"] = np.repeat(coarse, 3, axis=2).astype(np.uint8)

    grass = RNG.normal(0, 40, (H, W, 3))
    grass[:, :, 0] += 30; grass[:, :, 1] += 130; grass[:, :, 2] += 50
    out["green texture (grass)"] = grass.clip(0, 255).astype(np.uint8)

    stripes = np.zeros((H, W, 3), np.uint8)
    for x in range(0, W, 28):
        cv2.rectangle(stripes, (x, 0), (x + 14, H), (60, 90, 140), -1)
        cv2.rectangle(stripes, (x + 14, 0), (x + 28, H), (40, 70, 115), -1)
    out["striped (wood)"] = stripes

    check = np.zeros((H, W, 3), np.uint8)
    for y in range(0, H, 60):
        for x in range(0, W, 60):
            if (x // 60 + y // 60) % 2:
                cv2.rectangle(check, (x, y), (x + 60, y + 60), (200, 200, 200), -1)
    out["checkerboard"] = check
    return out


def draw_shapes(bg, gradient_fill):
    """Paint the five shapes, either flat or gradient filled."""
    img = bg.copy()
    fills = [(60, 90, 235), (235, 180, 40), (70, 200, 90),
             (200, 80, 210), (120, 60, 30)]
    for (kind, c, r), col in zip(SHAPES, fills):
        layer = np.zeros_like(img)
        mask = np.zeros(img.shape[:2], np.uint8)
        if kind == "circle":
            cv2.circle(mask, c, r, 255, -1)
        else:
            cv2.fillPoly(mask, [shape_points(kind, c, r)], 255)
        if gradient_fill:
            # a ramp between two very different colours, across the shape
            other = (col[2], col[0], col[1])
            ramp = np.zeros_like(img, np.float32)
            t = np.linspace(0, 1, img.shape[1], dtype=np.float32)[None, :, None]
            ramp[:] = np.float32(col) * (1 - t) + np.float32(other) * t
            layer = ramp.astype(np.uint8)
        else:
            layer[:] = col
        img[mask > 0] = layer[mask > 0]
    return img


def true_centres(shape_hw):
    """Exact area centroid of each drawn shape, measured from its own mask.

    Not the anchor point the shape was drawn from: a triangle's centroid sits a
    third of the way up from its base, so the anchor is 25 px out and scoring
    against it marks a perfectly correct detection wrong.
    """
    out = []
    for kind, c, r in SHAPES:
        m = np.zeros(shape_hw, np.uint8)
        if kind == "circle":
            cv2.circle(m, c, r, 255, -1)
        else:
            cv2.fillPoly(m, [shape_points(kind, c, r)], 255)
        mm = cv2.moments(m)
        out.append((kind, (mm["m10"] / mm["m00"], mm["m01"] / mm["m00"])))
    return out


def evaluate(img, tol=12):
    """Compare detections against the exact placement above."""
    dets, _, _ = da.detect(img)
    hits, wrong_class, dist_err = 0, 0, []
    used = set()
    for kind, c in true_centres(img.shape[:2]):
        best, bd = None, 1e9
        for i, d in enumerate(dets):
            if i in used:
                continue
            dd = np.hypot(d["center"][0] - c[0], d["center"][1] - c[1])
            if dd < bd:
                best, bd = i, dd
        if best is not None and bd <= tol:
            hits += 1
            used.add(best)
            dist_err.append(bd)
            if dets[best]["shape"] != kind:
                wrong_class += 1
    return {"found": hits, "expected": len(SHAPES), "detected": len(dets),
            "false_pos": len(dets) - hits, "wrong_class": wrong_class,
            "mean_centre_err": float(np.mean(dist_err)) if dist_err else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None, help="write a contact sheet here")
    args = ap.parse_args()

    tiles, rows = [], []
    print(f"{'background':<24} {'fill':<9} {'found':>7} {'false+':>7} "
          f"{'misclass':>9} {'centre err':>11}")
    print("-" * 72)
    tot = dict(found=0, exp=0, fp=0, wc=0, err=[])
    for name, bg in backgrounds().items():
        for grad in (False, True):
            img = draw_shapes(bg, grad)
            r = evaluate(img)
            rows.append((name, grad, r))
            tot["found"] += r["found"]; tot["exp"] += r["expected"]
            tot["fp"] += r["false_pos"]; tot["wc"] += r["wrong_class"]
            if r["found"]:
                tot["err"].append(r["mean_centre_err"])
            print(f"{name:<24} {'gradient' if grad else 'flat':<9} "
                  f"{r['found']}/{r['expected']:<5} {r['false_pos']:>7} "
                  f"{r['wrong_class']:>9} {r['mean_centre_err']:>10.1f}px")
            if args.save and not grad:
                vis = img.copy()
                for d in da.detect(img)[0]:
                    cv2.drawContours(vis, [d["contour"]], -1, (0, 255, 255), 2)
                    cv2.drawMarker(vis, d["center"], (0, 0, 255), cv2.MARKER_CROSS, 18, 2)
                t = cv2.resize(vis, (320, 180))
                cv2.putText(t, name, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 0, 255), 1, cv2.LINE_AA)
                tiles.append(t)

    print("-" * 72)
    print(f"{'TOTAL':<24} {'':<9} {tot['found']}/{tot['exp']:<5} {tot['fp']:>7} "
          f"{tot['wc']:>9} {np.mean(tot['err']):>10.1f}px")
    print(f"\nrecall {100 * tot['found'] / tot['exp']:.1f}%   "
          f"false positives {tot['fp']}   misclassified {tot['wc']}")

    if args.save and tiles:
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        cv2.imwrite(args.save, grid)
        print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
