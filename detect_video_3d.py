"""
Background-agnostic streaming detection, reporting metric (X, Y, Z).

Everything about the detector and the tracker is unchanged from
detect_video_agnostic -- this file adds one stage on the end, which converts the
tracked pixel centres into positions in the camera's frame. Keeping it a separate
stage matters: detection quality and calibration quality are then independently
testable, and the 3D layer cannot perturb a pipeline that already works.

The streaming contract survives intact. The scale memory (pose3d.PlaneScale)
reads the current frame's detections and its own record of past frames, never a
future one, so `python detect_video_3d.py 0` still runs a live webcam.

What the video adds over a single image is exactly what the scale needs. In one
photograph the circle must be present or there is no ruler at all. In a sequence,
the circle only has to appear *once*: while it is visible its depth also reveals
the true size of every other shape on the plane, and each of those becomes a
ruler in its own right. Frames where the circle is occluded, clipped or simply
out of view are still measured, and the overlay says which shape is doing the
measuring.

Usage:  python detect_video_3d.py [input] [-o out.mp4] [--csv log.csv]
                                  [--units in|ft|m] [--pp center]
"""

import argparse
import csv
import time

import cv2
import numpy as np

import detect_shapes_agnostic as ds
import detect_video_agnostic as dva
import pose3d

SRC_COLOR = {"circle": (120, 255, 120), "learned": (0, 210, 255),
             "held": (0, 160, 255)}


def measurable(track, frame_shape):
    """May this track's area be used as a ruler on this frame?

    Area is the entire measurement, so anything that hides part of a shape makes
    it read as further away than it is. Three ways that happens, all excluded:
    the track is coasting (its outline is stale), it is clipped by the frame
    edge, or something is in front of it. The last two are re-derived from the
    tracked contour with the same tests detect() applies to a fresh detection,
    which keeps this file from having to reach into the tracker's internals.
    """
    if track.misses > 0:
        return False
    h, w = frame_shape[:2]
    bx, by, bw, bh = cv2.boundingRect(track.contour)
    if bx <= 2 or by <= 2 or bx + bw >= w - 2 or by + bh >= h - 2:
        return False
    area = cv2.contourArea(track.contour)
    hull = max(cv2.contourArea(cv2.convexHull(track.contour)), 1.0)
    return area / hull >= 0.97


def draw_overlay(frame, tracks, frame_idx, fps, detections, Z, source, unit):
    """dva's overlay plus a second label line carrying the metric position."""
    out = frame.copy()
    h, w = out.shape[:2]
    s = w / 1920.0
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(round(2 * s)))

    for t in tracks:
        coasting = t.misses > 0
        cx, cy = int(round(t.center[0])), int(round(t.center[1]))

        pts = list(t.trail)
        for i in range(1, len(pts)):
            a = i / len(pts)
            cv2.line(out, pts[i - 1], pts[i],
                     tuple(int(c * a) for c in t.color), max(1, int(round(2 * s))))

        if not coasting:
            cv2.drawContours(out, [t.contour], -1, t.color, max(1, int(round(3 * s))))
        else:
            r = int(round(40 * s))
            for k in range(0, 360, 30):
                a0, a1 = np.deg2rad(k), np.deg2rad(k + 15)
                cv2.line(out, (int(cx + r * np.cos(a0)), int(cy + r * np.sin(a0))),
                         (int(cx + r * np.cos(a1)), int(cy + r * np.sin(a1))),
                         t.color, max(1, int(round(2 * s))))

        cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS,
                       int(round(26 * s)), thick)

        head = f"#{t.id} {t.label}"
        if coasting:
            head += " [predicted]"
        if getattr(t, "xyz", None):
            X, Y, Zc = pose3d.convert(t.xyz, unit)
            body = f"X{X:+.1f} Y{Y:+.1f} Z{Zc:.1f} {unit}"
        else:
            body = "no scale yet"

        lines = [(head, t.color), (body, SRC_COLOR.get(source, (200, 200, 200)))]
        fs = 0.58 * s
        sizes = [cv2.getTextSize(a, font, fs, thick)[0] for a, _ in lines]
        tw, th = max(a for a, _ in sizes), max(b for _, b in sizes)
        step = th + int(8 * s)
        bx, by, bw_, bh_ = cv2.boundingRect(t.contour)
        top = by - int(10 * s) - step
        if top - th < 0:
            top = by + bh_ + th + int(12 * s)
        tx = int(np.clip(bx + bw_ // 2 - tw // 2, 2, w - tw - 2))
        top = int(np.clip(top, th + int(8 * s), h - step - int(6 * s)))
        cv2.rectangle(out, (tx - 5, top - th - 6), (tx + tw + 5, top + step + 5),
                      (0, 0, 0), -1)
        for i, (text, col) in enumerate(lines):
            cv2.putText(out, text, (tx, top + i * step), font, fs, col,
                        thick, cv2.LINE_AA)

    depth = (f"depth {pose3d.convert((0, 0, Z), unit)[2]:6.1f} {unit} [{source}]"
             if Z else "depth   --   (circle not seen yet)")
    for line, y in zip([f"frame {frame_idx}", f"{fps:5.1f} fps",
                        f"tracked {len(tracks)}", f"detected {len(detections)}",
                        depth],
                       range(int(38 * s), int(38 * s) + 5 * int(34 * s), int(34 * s))):
        cv2.putText(out, line, (int(16 * s), y), font, 0.8 * s, (0, 0, 0),
                    thick + 2, cv2.LINE_AA)
        cv2.putText(out, line, (int(16 * s), y), font, 0.8 * s, (255, 255, 255),
                    thick, cv2.LINE_AA)
    return out


def run(source, output=None, csv_path=None, scale=1.0, max_frames=None,
        quiet=False, unit="in", principal="given", k_width=pose3d.K_REF_WIDTH):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"could not open {source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer, tracker, plane, rows = None, None, None, []
    frame_idx, proc_ms, counts, live_fps = 0, [], [], 0.0
    sources, depths = [], []

    while True:
        ok, frame = cap.read()          # <-- the only place a frame enters
        if not ok or (max_frames and frame_idx >= max_frames):
            break
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)

        t0 = time.perf_counter()
        detections, _, _ = ds.detect(frame)             # stateless, this frame only
        if tracker is None:
            tracker = dva.ShapeTracker(frame.shape)
            cam = pose3d.Camera(ref_width=k_width,
                                principal=principal).for_frame(frame.shape)
            plane = pose3d.PlaneScale(cam)              # causal, past frames only
        tracks = tracker.update(detections, frame_idx)

        # 3D stage. Track ids key the size memory, so a shape measured while the
        # circle was up stays a ruler under its own identity afterwards.
        obs = [(t.id, t.label, t.area, measurable(t, frame.shape),
                pose3d.circle_score(t.contour)) for t in tracks]
        Z, src = plane.update(obs)
        for t in tracks:
            t.xyz = plane.locate(t.center, Z) if Z else None

        proc_ms.append((time.perf_counter() - t0) * 1000)
        live_fps = 1000.0 / max(np.mean(proc_ms[-30:]), 1e-6)
        counts.append(len(tracks))
        sources.append(src)
        if Z:
            depths.append(Z)

        if csv_path:
            for t in tracks:
                row = t.as_row(frame_idx)
                X, Y, Zc = (pose3d.convert(t.xyz, unit) if t.xyz
                            else (None, None, None))
                row.update({f"X_{unit}": None if X is None else round(X, 2),
                            f"Y_{unit}": None if Y is None else round(Y, 2),
                            f"Z_{unit}": None if Zc is None else round(Zc, 2),
                            "depth_source": src or ""})
                rows.append(row)
        if output:
            vis = draw_overlay(frame, tracks, frame_idx, live_fps, detections,
                               Z, src, unit)
            if writer is None:
                writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"),
                                         src_fps, (vis.shape[1], vis.shape[0]))
            writer.write(vis)

        frame_idx += 1
        if not quiet and frame_idx % 100 == 0:
            z_txt = (f"{pose3d.convert((0, 0, Z), unit)[2]:6.1f} {unit}" if Z
                     else "   --  ")
            print(f"  frame {frame_idx}/{total}  {np.mean(proc_ms[-100:]):6.1f} ms  "
                  f"{live_fps:5.1f} fps  tracking {len(tracks)}  depth {z_txt}")

    cap.release()
    if writer:
        writer.release()
    if csv_path and rows:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return {"frames": frame_idx, "ms": proc_ms, "counts": counts, "rows": rows,
            "src_fps": src_fps, "sources": sources, "depths": depths,
            "tracks_created": dva.Track._next_id - 1}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", default="PennAir 2024 App Dynamic Hard.mp4")
    ap.add_argument("-o", "--output", default="output_hard_3d.mp4")
    ap.add_argument("--csv", default="track_log_3d.csv")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--units", default="in", choices=list(pose3d.UNITS))
    ap.add_argument("--pp", default="given", choices=["given", "center"],
                    help="principal point: 'given' uses K as supplied (cx=cy=0)")
    ap.add_argument("--k-width", type=float, default=pose3d.K_REF_WIDTH)
    args = ap.parse_args()

    source = int(args.input) if str(args.input).isdigit() else args.input
    print(f"streaming {source}")
    st = run(source, None if args.no_video else args.output, args.csv,
             args.scale, args.max_frames, unit=args.units,
             principal=args.pp, k_width=args.k_width)

    ms, counts = np.array(st["ms"]), np.array(st["counts"])
    u = args.units
    print(f"\n{st['frames']} frames processed one at a time")
    print(f"  per-frame  mean {ms.mean():6.1f} ms   median {np.median(ms):6.1f} ms   "
          f"p95 {np.percentile(ms, 95):6.1f} ms")
    print(f"  throughput {1000 / ms.mean():5.1f} fps   (source {st['src_fps']:.1f} fps)")
    print(f"  shapes tracked per frame: mean {counts.mean():.2f}  "
          f"min {counts.min()}  max {counts.max()}")

    if st["depths"]:
        d = np.array(st["depths"]) * pose3d.UNITS[u]
        n = len(st["sources"])
        got = {k: st["sources"].count(k) for k in ("circle", "learned", "held")}
        print(f"\n  depth on {sum(got.values())}/{n} frames "
              f"({100 * sum(got.values()) / n:.1f}%)")
        print(f"    from the circle       {got['circle']:5d} frames")
        print(f"    from a learned ruler  {got['learned']:5d} frames")
        print(f"    held (nothing usable) {got['held']:5d} frames")
        print(f"  plane depth  median {np.median(d):.2f} {u}   "
              f"spread {d.min():.2f} - {d.max():.2f} {u}   sd {d.std():.2f}")
    else:
        print("\n  no circle ever seen -> no metric scale established")

    if not args.no_video:
        print(f"  wrote {args.output}")
    if st["rows"]:
        print(f"  wrote {args.csv} ({len(st['rows'])} rows)")


if __name__ == "__main__":
    main()
