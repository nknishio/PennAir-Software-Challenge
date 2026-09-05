"""
Background-agnostic streaming shape detection.

Same streaming contract as detect_video.py -- frames arrive one at a time, the
detector is a pure function of the current frame, and the tracker reads only the
past -- but built on detect_shapes_agnostic, so it carries no assumption about
what the background looks like or how the shapes are filled.

One part of the tracker had to change. Identity was previously carried by each
shape's mean fill colour, which a gradient fill destroys: the mean of a
purple-to-green rectangle names a colour the shape does not contain, and it
drifts as the shape is occluded or clipped. Identity is now carried by a
hue/saturation histogram, which records which colours are present instead of
averaging them. Measured between consecutive frames, the same shape scores at
most 0.008 and two different shapes at least 0.893 -- a far wider margin than
mean colour ever gave.

Usage:  python detect_video_agnostic.py [input] [-o out.mp4] [--csv log.csv]
"""

import argparse
import csv
import time
from collections import Counter, deque

import cv2
import numpy as np

import detect_shapes_agnostic as ds

TRACK_COLORS = [(0, 255, 255), (255, 128, 0), (0, 200, 255), (255, 0, 255),
                (0, 255, 128), (255, 255, 0), (128, 0, 255), (0, 128, 255)]

HIST_CMP = cv2.HISTCMP_BHATTACHARYYA


def appearance_distance(a, b):
    """0 = identical, 1 = nothing in common. Missing descriptor = no opinion."""
    if a is None or b is None:
        return 0.5
    return float(cv2.compareHist(a, b, HIST_CMP))


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
        self.hist = det["appearance"]
        self.raw_label = det["shape"]
        self.contour = det["contour"]
        self.area = det["area"]
        self.votes = deque(maxlen=45)
        self.trail = deque(maxlen=trail_len)
        self.hits = 1
        self.misses = 0
        self.first_frame = frame_idx
        self.state = "confirmed" if self.is_strong(det, frame_area) else "new"
        self.color = TRACK_COLORS[(self.id - 1) % len(TRACK_COLORS)]
        self._record(det, frame_idx)

    def _record(self, det, frame_idx):
        # Only frames showing the whole shape may vote. A clipped or occluded
        # outline has the wrong vertex count by construction.
        if det is not None and not det["partial"] and not det["occluded"]:
            self.votes.append(det["shape"])
        self.trail.append((int(self.center[0]), int(self.center[1])))
        self.last_frame = frame_idx

    @property
    def label(self):
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
        measured = np.float32(det["center"])
        gain = 0.35 if (det["occluded"] or det["partial"]) else 0.8
        predicted = self.predict()
        new_center = predicted + gain * (measured - predicted)

        self.velocity = 0.5 * self.velocity + 0.5 * (new_center - self.center)
        self.center = new_center
        self.contour = det["contour"]
        self.area = det["area"]
        self.raw_label = det["shape"]
        # Drift the descriptor slowly, and only on a clean view, so a partly
        # hidden frame cannot rewrite what the shape is supposed to look like.
        if det["appearance"] is not None and not det["occluded"] and not det["partial"]:
            self.hist = (det["appearance"] if self.hist is None
                         else cv2.addWeighted(self.hist, 0.8, det["appearance"], 0.2, 0))
        self.hits += 1
        self.misses = 0
        if self.hits >= 3:
            self.state = "confirmed"
        self._record(det, frame_idx)

    def coast(self, frame_idx):
        self.center = self.center + self.velocity
        self.velocity *= 0.9
        self.misses += 1
        self._record(None, frame_idx)

    def as_row(self, frame_idx):
        return {"frame": frame_idx, "track_id": self.id, "shape": self.label,
                "cx": int(round(self.center[0])), "cy": int(round(self.center[1])),
                "area": int(self.area), "state": self.state,
                "confidence": round(self.confidence, 3)}


class ShapeTracker:
    """Nearest-neighbour tracker gated on position and appearance."""

    def __init__(self, frame_shape, gate_frac=0.06, min_hits=3, max_gap=20,
                 trail_len=48, appear_gate=0.6, appear_weight=60.0):
        self.h, self.w = frame_shape[:2]
        self.gate = gate_frac * frame_shape[1]
        self.min_hits = min_hits
        self.max_gap = max_gap
        self.trail_len = trail_len
        self.appear_gate = appear_gate
        self.appear_weight = appear_weight
        self.tracks = []

    def _in_frame(self, track, margin=60):
        x, y = track.center
        return (-margin <= x <= self.w + margin) and (-margin <= y <= self.h + margin)

    def update(self, detections, frame_idx):
        preds = [t.predict() for t in self.tracks]
        unmatched_d = set(range(len(detections)))
        pairs = []

        if self.tracks and detections:
            cost = np.full((len(self.tracks), len(detections)), np.inf, np.float32)
            for i, (p, t) in enumerate(zip(preds, self.tracks)):
                for j, d in enumerate(detections):
                    dist = np.linalg.norm(p - np.float32(d["center"]))
                    dapp = appearance_distance(t.hist, d["appearance"])
                    if dapp > self.appear_gate:
                        continue                     # a different shape entirely
                    # Appearance survives occlusion untouched while position
                    # lurches, so agreement on looks earns a wider positional gate.
                    limit = self.gate * (2.5 if dapp < 0.25 else 1.0)
                    if dist <= limit:
                        cost[i, j] = dist + self.appear_weight * dapp
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

    for line, y in zip([f"frame {frame_idx}", f"{fps:5.1f} fps",
                        f"tracked {len(tracks)}", f"detected {len(detections)}"],
                       range(int(38 * s), int(38 * s) + 4 * int(34 * s), int(34 * s))):
        cv2.putText(out, line, (int(16 * s), y), font, 0.8 * s, (0, 0, 0),
                    thick + 2, cv2.LINE_AA)
        cv2.putText(out, line, (int(16 * s), y), font, 0.8 * s, (255, 255, 255),
                    thick, cv2.LINE_AA)
    return out


def run(source, output=None, csv_path=None, scale=1.0, max_frames=None, quiet=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"could not open {source}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    writer, tracker, rows = None, None, []
    frame_idx, proc_ms, counts, live_fps = 0, [], [], 0.0

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
            tracker = ShapeTracker(frame.shape)
        tracks = tracker.update(detections, frame_idx)  # causal, past frames only
        proc_ms.append((time.perf_counter() - t0) * 1000)
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
            print(f"  frame {frame_idx}/{total}  {np.mean(proc_ms[-100:]):6.1f} ms  "
                  f"{live_fps:5.1f} fps  tracking {len(tracks)}")

    cap.release()
    if writer:
        writer.release()
    if csv_path and rows:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return {"frames": frame_idx, "ms": proc_ms, "counts": counts,
            "tracks_created": Track._next_id - 1, "src_fps": src_fps, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", default="PennAir 2024 App Dynamic Hard.mp4")
    ap.add_argument("-o", "--output", default="output_hard.mp4")
    ap.add_argument("--csv", default="track_log_hard.csv")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    source = int(args.input) if args.input.isdigit() else args.input
    print(f"streaming {source}")
    st = run(source, None if args.no_video else args.output, args.csv,
             args.scale, args.max_frames)

    ms, counts = np.array(st["ms"]), np.array(st["counts"])
    print(f"\n{st['frames']} frames processed one at a time")
    print(f"  per-frame  mean {ms.mean():6.1f} ms   median {np.median(ms):6.1f} ms   "
          f"p95 {np.percentile(ms, 95):6.1f} ms")
    print(f"  throughput {1000 / ms.mean():5.1f} fps   (source {st['src_fps']:.1f} fps)")
    print(f"  shapes tracked per frame: mean {counts.mean():.2f}  "
          f"min {counts.min()}  max {counts.max()}")
    print(f"  distinct tracks created: {st['tracks_created']}")
    if not args.no_video:
        print(f"  wrote {args.output}")
    if st["rows"]:
        print(f"  wrote {args.csv} ({len(st['rows'])} rows)")


if __name__ == "__main__":
    main()
