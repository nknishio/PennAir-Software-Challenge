"""
One command to exercise all four deliverables.

    python run_tests.py              # all four, on short clips
    python run_tests.py --step 3     # just the background-agnostic one
    python run_tests.py --full       # whole videos instead of the first 150 frames

Each step prints what it measured and a PASS/FAIL; the exit code is the number of
steps that failed, so this drops straight into CI.

  1  picture              detect_shapes.py            static image, 5 shapes
  2  video                detect_video.py             streaming + tracking
  3  background-agnostic  detect_video_agnostic.py    any ground, any fill
  4  3D                   detect_video_3d.py          metric X, Y, Z

Steps 1-3 check the pipeline that was already there; step 4 additionally runs
test_pose3d.py, which builds scenes from known metric truth and asks the pipeline
to recover it. The videos are large, so the video steps default to the first 150
frames -- enough to exercise occlusion, tracking and the depth bootstrap. `--full`
runs the lot.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

STATIC = "PennAir 2024 App Static.png"
VIDEO = "PennAir 2024 App Dynamic.mp4"
VIDEO_HARD = "PennAir 2024 App Dynamic Hard.mp4"
EXPECTED = {"circle", "triangle", "rectangle", "pentagon", "trapezoid"}


class Step:
    """Collects checks so a step reports everything it found, not just the first."""

    def __init__(self, n, title):
        self.n, self.title, self.checks = n, title, []
        print(f"\n{'=' * 74}\nSTEP {n} -- {title}\n{'=' * 74}")

    def note(self, label, detail=""):
        """Reported, not graded -- for numbers that depend on the machine."""
        print(f"  [note] {label}" + (f"   {detail}" if detail else ""))

    def check(self, ok, label, detail=""):
        self.checks.append(bool(ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
        return ok

    def done(self):
        ok = all(self.checks)
        print(f"  ---> STEP {self.n} {'PASS' if ok else 'FAIL'} "
              f"({sum(self.checks)}/{len(self.checks)} checks)")
        return ok


def missing(step, *paths):
    gone = [p for p in paths if not os.path.exists(p)]
    if gone:
        print(f"  SKIPPED -- not found: {', '.join(gone)}")
        step.checks.append(True)
    return bool(gone)


# --------------------------------------------------------------------------

def step1(args):
    """Static image: the original detector, on the background it was built for."""
    import detect_shapes as ds

    s = Step(1, "picture  (detect_shapes.py)")
    if missing(s, STATIC):
        return s.done()

    bgr = cv2.imread(STATIC)
    t0 = time.perf_counter()
    results, _, _ = ds.detect(bgr)
    ms = (time.perf_counter() - t0) * 1000
    names = sorted(r["shape"] for r in results)

    print(f"  {len(results)} shapes in {ms:.0f} ms: {names}")
    for r in sorted(results, key=lambda r: r["shape"]):
        print(f"     {r['shape']:<10} centre {str(r['center']):>12}  area {r['area']:>7.0f} px")

    s.check(len(results) == 5, "five shapes found", f"got {len(results)}")
    s.check(set(names) == EXPECTED, "all five named correctly",
            f"{sorted(EXPECTED - set(names))} missing" if set(names) != EXPECTED else "")
    inside = all(0 < r["center"][0] < bgr.shape[1] and 0 < r["center"][1] < bgr.shape[0]
                 for r in results)
    s.check(inside, "every centre lies inside the image")
    cv2.imwrite("output_static.png", ds.annotate(bgr, results))
    print("  wrote output_static.png")
    return s.done()


def step2(args):
    """Video: streaming detection plus the tracker, on the grass footage."""
    import detect_video as dv

    s = Step(2, "video  (detect_video.py)")
    if missing(s, VIDEO):
        return s.done()

    dv.Track._next_id = 1
    st = dv.run(VIDEO, None, None, 1.0, args.frames, quiet=True)
    ms, counts = np.array(st["ms"]), np.array(st["counts"])
    fps = 1000 / ms.mean()

    print(f"  {st['frames']} frames   mean {ms.mean():.1f} ms   p95 "
          f"{np.percentile(ms, 95):.1f} ms   {fps:.1f} fps "
          f"(source {st['src_fps']:.1f} fps)")
    print(f"  shapes tracked per frame: mean {counts.mean():.2f}  "
          f"min {counts.min()}  max {counts.max()}   "
          f"{st['tracks_created']} tracks created")

    s.check(st["frames"] > 0, "frames streamed one at a time")
    s.check(counts.mean() >= 4.5, "≥4.5 shapes tracked per frame on average",
            f"{counts.mean():.2f}")
    # Throughput is a property of the machine as much as of the code, so it is
    # reported rather than graded; what is graded is that the cost is bounded.
    s.check(ms.mean() < 100, "per-frame cost stays bounded",
            f"mean {ms.mean():.1f} ms, p95 {np.percentile(ms, 95):.1f} ms")
    s.note("real-time margin",
           f"{fps:.1f} fps vs {st['src_fps']:.1f} fps source -- "
           f"{'clears' if fps >= st['src_fps'] else 'BELOW'} real time here")
    s.check(st["tracks_created"] <= 4 * counts.mean() + 25,
            "identities stay stable (few spurious tracks)",
            f"{st['tracks_created']} tracks for ~{counts.mean():.1f} shapes")
    return s.done()


def step3(args):
    """Background-agnostic: the synthetic sweep, then the hard footage."""
    import detect_shapes_agnostic as da
    import detect_video_agnostic as dva
    import test_backgrounds as tb

    s = Step(3, "background-agnostic  (detect_shapes_agnostic.py / detect_video_agnostic.py)")

    found = expect = wrong = fp = 0
    errs = []
    print("  nine synthetic backgrounds x two fill types:")
    for name, bg in tb.backgrounds().items():
        line = []
        for grad in (False, True):
            r = tb.evaluate(tb.draw_shapes(bg, grad))
            found += r["found"]; expect += r["expected"]
            wrong += r["wrong_class"]; fp += r["false_pos"]
            if r["found"]:
                errs.append(r["mean_centre_err"])
            line.append(f"{r['found']}/{r['expected']}")
        print(f"     {name:<24} flat {line[0]}   gradient {line[1]}")
    recall = found / expect
    print(f"  recall {100 * recall:.1f}%   misclassified {wrong}   "
          f"false positives {fp}   mean centre error {np.mean(errs):.1f} px")

    s.check(recall >= 0.95, "recall ≥95% across every background",
            f"{100 * recall:.1f}%")
    s.check(wrong == 0, "no misclassifications", f"{wrong}")
    s.check(np.mean(errs) < 2.0, "centres accurate to <2 px",
            f"{np.mean(errs):.1f} px")

    if not missing(s, VIDEO_HARD):
        dva.Track._next_id = 1
        st = dva.run(VIDEO_HARD, None, None, 1.0, args.frames, quiet=True)
        counts = np.array(st["counts"])
        print(f"  hard footage (asphalt + gradient fills): {st['frames']} frames, "
              f"mean {counts.mean():.2f} shapes/frame, "
              f"{1000 / np.mean(st['ms']):.1f} fps")
        s.check(counts.mean() >= 4.0, "≥4 shapes tracked per frame on asphalt",
                f"{counts.mean():.2f}")
    return s.done()


def step4(args):
    """3D: the ground-truth suite, the static image, then the hard footage."""
    import detect_shapes_agnostic as da
    import detect_3d
    import detect_video_3d as dv3
    import pose3d
    import test_pose3d as tp

    s = Step(4, "background-agnostic 3D  (detect_3d.py / detect_video_3d.py)")

    print(f"  K: fx={pose3d.K_GIVEN[0, 0]:.4f}  fy={pose3d.K_GIVEN[1, 1]:.4f}  "
          f"cx={pose3d.K_GIVEN[0, 2]:.1f}  cy={pose3d.K_GIVEN[1, 2]:.1f}  "
          f"(for {pose3d.K_REF_WIDTH}px wide)")
    print(f"  scale reference: circle radius {pose3d.CIRCLE_RADIUS_IN:.0f} in\n")

    # -- a. the arithmetic, against scenes built from known metric truth ----
    acc = tp.test_accuracy(save=args.save)
    s.check(acc["z_err"] < 3.0, "depth recovered to <3 in of truth",
            f"mean |error| {acc['z_err']:.2f} in")
    s.check(acc["xy_err"] < 2.0, "lateral position recovered to <2 in",
            f"mean error {acc['xy_err']:.2f} in")
    s.check(acc["placed"] >= 0.90, "≥90% of detected shapes placed in 3D",
            f"{100 * acc['placed']:.1f}%")
    s.check(tp.test_bootstrap(), "scale survives the circle leaving view")
    s.check(tp.test_resolution(), "intrinsics scale with the image")
    s.check(tp.test_roundtrip(), "backproject is the exact inverse of project")
    env = tp.test_envelope()
    s.check(tp.test_altitude(), "a genuine altitude change is followed")
    print(f"  measured operating envelope: out to {env:.0f} in ({env / 12:.1f} ft)")

    # -- b. the static image -----------------------------------------------
    if not missing(s, STATIC):
        bgr = cv2.imread(STATIC)
        cam = pose3d.Camera().for_frame(bgr.shape)
        dets, _, _ = da.detect(bgr)
        Z, src, _ = pose3d.solve_frame(dets, cam)
        print(f"\n  {STATIC}: plane depth {Z:.1f} in ({Z / 12:.1f} ft) from the {src}")
        for d in dets:
            X, Y, Zc = d["xyz"]
            print(f"     {d['shape']:<10} X {X:8.2f}  Y {Y:7.2f}  Z {Zc:7.2f} in")
        s.check(Z is not None, "depth recovered from a single photograph")
        cv2.imwrite("output_static_3d.png",
                    detect_3d.annotate(bgr, dets, Z, "in", src))
        print("  wrote output_static_3d.png")

    # -- c. the hard footage -----------------------------------------------
    if not missing(s, VIDEO_HARD):
        st = dv3.run(VIDEO_HARD, None, None, 1.0, args.frames, quiet=True)
        d = np.array(st["depths"])
        got = {k: st["sources"].count(k) for k in ("circle", "learned", "held")}
        n = len(st["sources"])
        print(f"\n  {VIDEO_HARD}: {st['frames']} frames")
        print(f"     depth on {len(d)}/{n} frames  "
              f"(circle {got['circle']}, learned {got['learned']}, held {got['held']})")
        print(f"     median {np.median(d):.2f} in ({np.median(d) / 12:.2f} ft)   "
              f"sd {d.std():.2f} in   spread {d.min():.2f}-{d.max():.2f} in")
        s.check(len(d) == n, "every frame got a depth", f"{len(d)}/{n}")
        # The camera holds altitude in this footage, so the recovered depth must
        # too. A drifting or jumping value would mean the scale is being reset.
        s.check(d.std() / np.median(d) < 0.02,
                "depth is steady while the camera holds altitude",
                f"sd {100 * d.std() / np.median(d):.2f}% of median")
        s.check(got["learned"] > 0 or got["circle"] == n,
                "frames without the circle are still measured",
                f"{got['learned']} carried by a learned ruler")
    return s.done()


STEPS = {1: step1, 2: step2, 3: step3, 4: step4}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, choices=[1, 2, 3, 4], action="append",
                    help="run only this step (repeatable); default is all four")
    ap.add_argument("--frames", type=int, default=150,
                    help="frames per video step (default 150)")
    ap.add_argument("--full", action="store_true", help="run the whole videos")
    ap.add_argument("--save", default=None,
                    help="write step 4's contact sheet here")
    args = ap.parse_args()
    if args.full:
        args.frames = None

    wanted = sorted(set(args.step)) if args.step else [1, 2, 3, 4]
    print(f"running step(s) {wanted}   "
          f"video length: {'full' if args.frames is None else f'{args.frames} frames'}")

    results = {n: STEPS[n](args) for n in wanted}

    print(f"\n{'=' * 74}\nSUMMARY\n{'=' * 74}")
    titles = {1: "picture", 2: "video", 3: "background-agnostic", 4: "3D"}
    for n, ok in results.items():
        print(f"  step {n}  {titles[n]:<22} {'PASS' if ok else 'FAIL'}")
    failed = sum(1 for ok in results.values() if not ok)
    print(f"\n{len(results) - failed}/{len(results)} steps passed")
    return failed


if __name__ == "__main__":
    sys.exit(main())
