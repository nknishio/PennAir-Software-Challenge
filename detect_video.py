"""
Streaming shape detection for video -- one frame in, one result out.

The detector in detect_shapes.py is a pure function of a single frame, which is
what makes it usable on a live feed: this script pulls frames off the stream one
at a time, and nothing in the pipeline ever looks at a frame that has not
arrived yet. No seeking, no buffering, no second pass. Run it against a camera
index instead of a file and it behaves identically.

A single frame is not enough on its own, though. Shapes overlap, and while a
shape is behind another one the geometry that identifies it is simply not in the
image. So this adds the one thing a video has that a photo does not -- memory:

    detect (per frame, stateless)  ->  track (across frames, causal)

The tracker carries identity, votes on the label over time so a moment of
occlusion cannot rename a shape, and coasts on a motion model when a shape is
lost entirely. All of it uses only past frames.

Usage:  python detect_video.py [input] [-o out.mp4] [--csv log.csv] [--scale 0.5]
"""

import argparse
import csv
import time
from collections import Counter, deque

import cv2
import numpy as np

import detect_shapes as ds

# Distinct per-track colours (BGR), reused cyclically.
TRACK_COLORS = [(0, 255, 255), (255, 128, 0), (0, 200, 255), (255, 0, 255),
                (0, 255, 128), (255, 255, 0), (128, 0, 255), (0, 128, 255)]


class Track:
    """One shape followed through time."""

    _next_id = 1

    @staticmethod
    def is_strong(det, frame_area):
        return (not det["partial"] and not det["occluded"]
                and det["area"] >= 0.002 * frame_area)

    def __init__(self, det, frame_idx, trail_len, frame_area):
        self.id = Track._next_id
        Track._next_id += 1
        self.center = np.float32(det["center"])
        self.velocity = np.zeros(2, np.float32)
        self.fill = np.float32(det["color"])   # identity cue: each shape is one flat colour
        self.raw_label = det["shape"]
        self.contour = det["contour"]
        self.area = det["area"]
        self.votes = deque(maxlen=45)          # ~1.5 s of label evidence
        self.trail = deque(maxlen=trail_len)
        self.hits = 1
        self.misses = 0
        self.first_frame = frame_idx
        # The confirmation delay exists to reject flicker, but a large, solid,
        # fully-visible shape is not flicker -- making it wait three frames just
        # loses the opening of the stream. Believe those immediately; make
        # marginal blobs earn it.
        self.state = "confirmed" if self.is_strong(det, frame_area) else "new"
        self.color = TRACK_COLORS[(self.id - 1) % len(TRACK_COLORS)]
        self._record(det, frame_idx)

    def _record(self, det, frame_idx):
        # Only vote from frames where the whole shape is actually visible. A
        # clipped or occluded outline has the wrong vertex count by construction,
        # so letting it vote would be letting noise rename the shape.
        if det is not None and not det["partial"] and not det["occluded"]:
            self.votes.append(det["shape"])
        self.trail.append((int(self.center[0]), int(self.center[1])))
        self.last_frame = frame_idx

    @property
    def label(self):
        """Majority label over recent clean frames.

        A shape that has been occluded or clipped for its whole life so far has
        no clean votes; fall back to the latest raw reading rather than refusing
        to name it.
        """
        if self.votes:
            return Counter(self.votes).most_common(1)[0][0]
        return self.raw_label or "unknown"

    @property
    def confidence(self):
        if not self.votes:
            return 0.0
        return Counter(self.votes).most_common(1)[0][1] / len(self.votes)

    def predict(self):
        return self.center + self.velocity

    def update(self, det, frame_idx):
        """Fold in a matched detection."""
        measured = np.float32(det["center"])

        # Trust the measurement when the shape is cleanly visible. When it is
        # occluded or half out of frame its centroid is biased, so lean on the
        # motion model instead of letting the marker jump.
        gain = 0.35 if (det["occluded"] or det["partial"]) else 0.8
        predicted = self.predict()
        new_center = predicted + gain * (measured - predicted)

        self.velocity = 0.5 * self.velocity + 0.5 * (new_center - self.center)
        self.center = new_center
        self.contour = det["contour"]
        self.area = det["area"]
        self.raw_label = det["shape"]
        self.fill = 0.7 * self.fill + 0.3 * np.float32(det["color"])
        self.hits += 1
        self.misses = 0
        if self.hits >= 3:
            self.state = "confirmed"
        self._record(det, frame_idx)

    def coast(self, frame_idx):
        """No detection this frame -- carry on under the motion model."""
        self.center = self.center + self.velocity
        self.velocity *= 0.9                   # bleed off speed; don't drift forever
        self.misses += 1
        self._record(None, frame_idx)

    def as_row(self, frame_idx):
        return {"frame": frame_idx, "track_id": self.id, "shape": self.label,
                "cx": int(round(self.center[0])), "cy": int(round(self.center[1])),
                "area": int(self.area), "state": self.state,
                "confidence": round(self.confidence, 3)}


class ShapeTracker:
    """Nearest-neighbour tracker over detection centroids.

    Deliberately simple: the shapes move smoothly and never swap places, so a
    gated greedy match on predicted position is enough, and it costs microseconds
    per frame. The parts that matter for a live feed are the gate (a match must
    be plausible), the confirmation delay (a blob must persist to become a
    shape), and the coast window (a shape may vanish briefly and come back as
    itself).
    """

    def __init__(self, frame_shape, gate_frac=0.06, min_hits=3, max_gap=20,
                 trail_len=48, color_gate=70.0, color_weight=0.5):
        self.h, self.w = frame_shape[:2]
        self.gate = gate_frac * frame_shape[1]     # px of allowed jump per frame
        self.color_gate = color_gate
        self.color_weight = color_weight
        self.min_hits = min_hits
        self.max_gap = max_gap
        self.trail_len = trail_len
        self.tracks = []

    def _in_frame(self, track, margin=60):
        x, y = track.center
        return (-margin <= x <= self.w + margin) and (-margin <= y <= self.h + margin)

    def update(self, detections, frame_idx):
        preds = [t.predict() for t in self.tracks]
        unmatched_d = set(range(len(detections)))
        pairs = []

        # Greedy: repeatedly commit the closest surviving pair. With a handful of
        # objects this matches what the Hungarian algorithm would give, without
        # the dependency.
        #
        # Position alone is not enough. When a shape is occluded its centroid
        # lurches, and if that lurch exceeds the gate the tracker drops it and
        # starts a new track -- the same shape, a new identity. But every shape
        # is one flat colour, and colour survives occlusion untouched. Matching
        # on position *and* colour keeps hold of a shape through the exact moment
        # position becomes unreliable, and stops two nearby shapes swapping ids.
        if self.tracks and detections:
            cost = np.full((len(self.tracks), len(detections)), np.inf, np.float32)
            for i, (p, t) in enumerate(zip(preds, self.tracks)):
                for j, d in enumerate(detections):
                    dist = np.linalg.norm(p - np.float32(d["center"]))
                    dcol = np.linalg.norm(t.fill - np.float32(d["color"]))
                    if dcol > self.color_gate:
                        continue                      # a different shape entirely
                    # A colour match earns a wider positional gate, since the
                    # jump is then far more likely to be occlusion than a mix-up.
                    limit = self.gate * (2.5 if dcol < 40 else 1.0)
                    if dist <= limit:
                        cost[i, j] = dist + self.color_weight * dcol
            while True:
                i, j = np.unravel_index(np.argmin(cost), cost.shape)
                if not np.isfinite(cost[i, j]):
                    break
                pairs.append((i, j))
                cost[i, :] = np.inf
                cost[:, j] = np.inf

        matched_t = set()
        for i, j in pairs:
            self.tracks[i].update(detections[j], frame_idx)
            matched_t.add(i)
            unmatched_d.discard(j)

        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.coast(frame_idx)
                if t.state == "confirmed":
                    t.state = "coasting"

        for j in sorted(unmatched_d):
            self.tracks.append(Track(detections[j], frame_idx, self.trail_len,
                                     self.w * self.h))

        # Retire anything unseen for too long, and any new blob that never
        # persisted long enough to be believed. Also stop coasting a shape once
        # its predicted position has left the frame -- out of view is out of
        # scope, and a phantom drifting off-screen is worse than no track.
        self.tracks = [t for t in self.tracks
                       if t.misses <= self.max_gap
                       and not (t.state == "new" and t.misses > 2)
                       and self._in_frame(t)]

        for t in self.tracks:
            if t.state == "coasting" and t.misses == 0:
                t.state = "confirmed"
        return [t for t in self.tracks
                if t.hits >= self.min_hits or t.state in ("confirmed", "coasting")]


def draw_overlay(frame, tracks, frame_idx, fps, detections):
    out = frame.copy()
    h, w = out.shape[:2]
    s = w / 1920.0                                  # scale annotation with resolution
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(round(2 * s)))

    for t in tracks:
        coasting = t.misses > 0
        cx, cy = int(round(t.center[0])), int(round(t.center[1]))

        # Trail: where this shape has been, fading with age.
        pts = list(t.trail)
        for i in range(1, len(pts)):
            a = i / len(pts)
            cv2.line(out, pts[i - 1], pts[i],
                     tuple(int(c * a) for c in t.color), max(1, int(round(2 * s))))

        if not coasting:
            cv2.drawContours(out, [t.contour], -1, t.color, max(1, int(round(3 * s))))
        else:
            # Predicted only -- show a dashed box so it never reads as a measurement.
            r = int(round(40 * s))
            for k in range(0, 360, 30):
                a0, a1 = np.deg2rad(k), np.deg2rad(k + 15)
                cv2.line(out, (int(cx + r * np.cos(a0)), int(cy + r * np.sin(a0))),
                         (int(cx + r * np.cos(a1)), int(cy + r * np.sin(a1))),
                         t.color, max(1, int(round(2 * s))))

        cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS,
                       int(round(26 * s)), thick)

        label = f"#{t.id} {t.label} ({cx},{cy})"
        if coasting:
            label += " [predicted]"
        fs = 0.62 * s
        (tw, th), _ = cv2.getTextSize(label, font, fs, thick)
        bx, by, bw_, bh_ = cv2.boundingRect(t.contour)
        ty = by - int(10 * s) if by - th - int(14 * s) >= 0 else by + bh_ + th + int(12 * s)
        tx = int(np.clip(bx + bw_ // 2 - tw // 2, 2, w - tw - 2))
        ty = int(np.clip(ty, th + int(8 * s), h - 4))
        cv2.rectangle(out, (tx - 5, ty - th - 6), (tx + tw + 5, ty + 5), (0, 0, 0), -1)
        cv2.putText(out, label, (tx, ty), font, fs, t.color, thick, cv2.LINE_AA)

    hud = [f"frame {frame_idx}", f"{fps:5.1f} fps", f"tracked {len(tracks)}",
           f"detected {len(detections)}"]
    y = int(38 * s)
    for line in hud:
        cv2.putText(out, line, (int(16 * s), y), font, 0.8 * s, (0, 0, 0),
                    thick + 2, cv2.LINE_AA)
        cv2.putText(out, line, (int(16 * s), y), font, 0.8 * s, (255, 255, 255),
                    thick, cv2.LINE_AA)
        y += int(34 * s)
    return out


def run(source, output=None, csv_path=None, scale=1.0, max_frames=None, quiet=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"could not open {source}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer, tracker, rows = None, None, []
    frame_idx = 0
    proc_ms = []
    live_fps = 0.0
    counts = []

    while True:
        ok, frame = cap.read()          # <-- the only place a frame enters. One at a time.
        if not ok:
            break
        if max_frames and frame_idx >= max_frames:
            break

        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)

        t0 = time.perf_counter()
        detections, _, _ = ds.detect(frame)             # stateless, this frame only
        if tracker is None:
            tracker = ShapeTracker(frame.shape)
        tracks = tracker.update(detections, frame_idx)  # causal, past frames only
        dt = (time.perf_counter() - t0) * 1000
        proc_ms.append(dt)
        live_fps = 1000.0 / max(np.mean(proc_ms[-30:]), 1e-6)
        counts.append(len(tracks))

        if csv_path:
            rows.extend(t.as_row(frame_idx) for t in tracks)

        if output:
            vis = draw_overlay(frame, tracks, frame_idx, live_fps, detections)
            if writer is None:
                writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"),
                                         src_fps, (vis.shape[1], vis.shape[0]))
            writer.write(vis)

        frame_idx += 1
        if not quiet and frame_idx % 100 == 0:
            print(f"  frame {frame_idx}/{total}  {np.mean(proc_ms[-100:]):5.1f} ms  "
                  f"{live_fps:5.1f} fps  tracking {len(tracks)}")

    cap.release()
    if writer:
        writer.release()

    if csv_path and rows:
        with open(csv_path, "w", newline="") as fh:
            wcsv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(rows)

    return {"frames": frame_idx, "ms": proc_ms, "counts": counts,
            "tracks_created": Track._next_id - 1, "src_fps": src_fps,
            "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", default="PennAir 2024 App Dynamic.mp4",
                    help="video file, or a camera index like 0")
    ap.add_argument("-o", "--output", default="output_dynamic.mp4")
    ap.add_argument("--csv", default="track_log.csv")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="process at this fraction of native resolution")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true", help="measure only, write no video")
    args = ap.parse_args()

    source = int(args.input) if args.input.isdigit() else args.input
    print(f"streaming {source}")
    stats = run(source, None if args.no_video else args.output, args.csv,
                args.scale, args.max_frames)

    ms = np.array(stats["ms"])
    counts = np.array(stats["counts"])
    print(f"\n{stats['frames']} frames processed one at a time")
    print(f"  per-frame  mean {ms.mean():5.1f} ms   median {np.median(ms):5.1f} ms   "
          f"p95 {np.percentile(ms, 95):5.1f} ms   max {ms.max():5.1f} ms")
    print(f"  throughput {1000 / ms.mean():5.1f} fps   (source is {stats['src_fps']:.1f} fps"
          f" -> {'real-time capable' if 1000 / ms.mean() >= stats['src_fps'] else 'BELOW real-time'})")
    print(f"  shapes tracked per frame: mean {counts.mean():.2f}  "
          f"min {counts.min()}  max {counts.max()}")
    print(f"  distinct tracks created: {stats['tracks_created']}")
    if not args.no_video:
        print(f"  wrote {args.output}")
    if stats["rows"]:
        print(f"  wrote {args.csv} ({len(stats['rows'])} rows)")


if __name__ == "__main__":
    main()
