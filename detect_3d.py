"""
Static image, in three dimensions.

detect_shapes_agnostic finds the shapes and their pixel centres; pose3d turns
each centre into a metric (X, Y, Z) in the camera's frame, using the supplied
intrinsics and the circle's known 10 in radius as the only scale reference.

The plane is assumed flat and fronto-parallel, so one depth covers every shape,
and that depth comes from the circle. A single image therefore has to contain the
circle -- there is no earlier frame to have learned a size from. The video
version (detect_video_3d.py) does not have that restriction.

Usage:  python detect_3d.py [image] [-o out.png] [--units in|ft|m] [--pp center]
"""

import argparse

import cv2
import numpy as np

import detect_shapes_agnostic as ds
import pose3d


def annotate(bgr, results, Z, unit="in", source="circle"):
    """Outline, centre, and the metric coordinates of that centre."""
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

        lines = [r["shape"]]
        if r.get("xyz"):
            X, Y, Zc = pose3d.convert(r["xyz"], unit)
            lines.append(f"X{X:+.1f} Y{Y:+.1f} Z{Zc:.1f}{unit}")
        else:
            lines.append("no scale")

        fs = 0.52 * s
        sizes = [cv2.getTextSize(t, font, fs, thick)[0] for t in lines]
        tw, th = max(a for a, _ in sizes), max(b for _, b in sizes)
        step = th + int(7 * s)
        bx, by, bw, bh = cv2.boundingRect(r["contour"])
        top = by - int(8 * s) - step * (len(lines) - 1)
        if top - th < 0:
            top = by + bh + th + int(10 * s)
        tx = int(np.clip(bx + bw // 2 - tw // 2, 2, w - tw - 2))
        top = int(np.clip(top, th + int(6 * s), h - step * len(lines)))
        cv2.rectangle(out, (tx - 4, top - th - 5),
                      (tx + tw + 4, top + step * (len(lines) - 1) + 5), (0, 0, 0), -1)
        for i, t in enumerate(lines):
            cv2.putText(out, t, (tx, top + i * step), font, fs,
                        (255, 255, 255) if i == 0 else (120, 255, 120),
                        thick, cv2.LINE_AA)

    if Z:
        # Along the bottom, not the top: a shape near the top-left corner has its
        # own label pushed to the very edge, and the two collide there.
        hud = f"plane depth {pose3d.convert((0, 0, Z), unit)[2]:.1f} {unit}  (from {source})"
        fs = 0.62 * s
        (tw, th), _ = cv2.getTextSize(hud, font, fs, thick)
        x0, y0 = int(14 * s), h - int(14 * s)
        cv2.rectangle(out, (x0 - 6, y0 - th - 8), (x0 + tw + 6, y0 + 6), (0, 0, 0), -1)
        cv2.putText(out, hud, (x0, y0), font, fs, (255, 255, 255), thick, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", nargs="?", default="PennAir 2024 App Static.png")
    ap.add_argument("-o", "--output", default="output_static_3d.png")
    ap.add_argument("--units", default="in", choices=list(pose3d.UNITS))
    ap.add_argument("--pp", default="given", choices=["given", "center"],
                    help="principal point: 'given' uses K exactly as supplied "
                         "(cx=cy=0); 'center' puts it at the image centre")
    ap.add_argument("--k-width", type=float, default=pose3d.K_REF_WIDTH,
                    help="image width the calibration was made at")
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"could not read {args.image}")

    cam = pose3d.Camera(ref_width=args.k_width,
                        principal=args.pp).for_frame(bgr.shape)
    results, _, _ = ds.detect(bgr)
    Z, source, _ = pose3d.solve_frame(results, cam)

    cv2.imwrite(args.output, annotate(bgr, results, Z, args.units, source))

    u = args.units
    print(f"{len(results)} shapes in {args.image}  ({bgr.shape[1]}x{bgr.shape[0]})")
    print(f"  fx={cam.fx:.1f}  fy={cam.fy:.1f}  cx={cam.cx:.1f}  cy={cam.cy:.1f}"
          f"   principal point: {args.pp}")
    if Z is None:
        print("  no circle found -> no metric scale; pixel centres only")
    else:
        print(f"  plane depth {pose3d.convert((0, 0, Z), u)[2]:.2f} {u} "
              f"({Z / 12:.2f} ft) from the {source}\n")
        print(f"  {'shape':<10} {'u':>5} {'v':>5} {'X':>9} {'Y':>9} {'Z':>9}"
              f"   sqrt(area) {u}")
        for r in results:
            X, Y, Zc = pose3d.convert(r["xyz"], u)
            side = np.sqrt(cam.metric_area(r["area"], Z)) * pose3d.UNITS[u]
            print(f"  {r['shape']:<10} {r['center'][0]:>5} {r['center'][1]:>5} "
                  f"{X:>9.2f} {Y:>9.2f} {Zc:>9.2f}   {side:>10.2f}")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
