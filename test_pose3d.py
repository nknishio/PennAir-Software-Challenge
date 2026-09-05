"""
Ground-truth test for the 3D stage.

Real footage cannot test this. Nobody measured how far the drone was from the
ground when it was shot, so the only thing the supplied videos can show is that
the depth comes out *stable* -- which a constant-valued bug would also achieve.

So the scene is built the other way round. Shapes are defined in inches on a
plane at a chosen depth and projected through the camera to make the image, then
the pipeline is asked to recover what was put in. Ground truth is exact because
it is the input.

The five checks:

  1. depth        does Z come back right, over a 12x range of distances and
                  several backgrounds and fill types
  2. position     do X and Y come back right, across the whole frame
  3. bootstrap    with the circle removed from view, does a shape it previously
                  measured carry the scale
  4. resolution   does the same scene at 1920 and at 960 give the same answer,
                  i.e. are the intrinsics scaled with the image
  5. round-trip   is backproject the exact inverse of project
  6. envelope     over what range of distances does any of this hold
  7. altitude     a real climb is followed, while a one-frame spike is not

Usage:  python test_pose3d.py [--save contact_sheet.png]
"""

import argparse

import cv2
import numpy as np

import detect_shapes_agnostic as da
import pose3d

# metric truth, inches. Anchors are fractions of the frame so the layout holds
# at any depth; the circle's radius is the 10 in the brief gives us.
SHAPES = [
    ("circle",    (0.55, 0.18), 10.0),
    ("triangle",  (0.30, 0.68), 13.0),
    ("rectangle", (0.18, 0.24), 11.0),
    ("pentagon",  (0.73, 0.72), 12.0),
    ("trapezoid", (0.88, 0.40), 12.5),
]

FILLS = [(60, 90, 235), (70, 200, 90), (235, 180, 40), (200, 80, 210), (120, 60, 30)]


def metric_points(kind, cx, cy, r):
    """Shape outline in inches on the plane. The circle is drawn separately."""
    if kind == "triangle":
        return np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.float64)
    if kind == "rectangle":
        return np.array([[cx - r, cy - 0.78 * r], [cx + r, cy - 0.78 * r],
                         [cx + r, cy + 0.78 * r], [cx - r, cy + 0.78 * r]], np.float64)
    if kind == "pentagon":
        a = np.linspace(-np.pi / 2, 3 * np.pi / 2, 6)[:5]
        return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)
    if kind == "trapezoid":
        return np.array([[cx - 0.5 * r, cy - r], [cx + 0.5 * r, cy - r],
                         [cx + r, cy + r], [cx - r, cy + r]], np.float64)
    return None


def scene_fits(cam, Z, w, h):
    """Does every shape land wholly inside the frame at this depth?

    Close in, the shapes are large enough to run off the edge of the picture, and
    a clipped shape shows less area than it has -- which the pipeline correctly
    refuses to measure. That is a property of this test's layout, not of the
    method, so the sweep below says so rather than scoring it as a failure.
    """
    for kind, (fx_, fy_), r in SHAPES:
        px, py = fx_ * w, fy_ * h
        ax, ay = cam.pixel_radius(r, Z)
        if px - ax < 0 or px + ax > w or py - ay < 0 or py + ay > h:
            return False
    return True


def backgrounds(w, h, seed=11):
    """Reseeded per call, so every run produces the same scenes and the numbers
    quoted in the README reproduce exactly rather than approximately."""
    rng = np.random.default_rng(seed)
    out = {}
    out["solid grey"] = np.full((h, w, 3), 118, np.uint8)
    g = rng.normal(0, 40, (h, w, 3))
    g[:, :, 0] += 30; g[:, :, 1] += 130; g[:, :, 2] += 50
    out["grass"] = g.clip(0, 255).astype(np.uint8)
    a = rng.normal(64, 26, (h // 2, w // 2, 1)).clip(0, 255)
    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
    out["asphalt"] = np.repeat(a, 3, axis=2).astype(np.uint8)
    return out


def render(cam, Z, w, h, bg, gradient=False, omit=()):
    """Project the metric scene onto the image, and report its exact truth.

    Truth is read back from each shape's own rendered mask rather than from the
    anchor it was drawn around: a triangle's centroid sits a third of the way up
    from its base. Because the plane is fronto-parallel, projection is an exact
    affine map, so back-projecting the pixel centroid of the mask gives the
    metric centroid of the shape with no approximation.
    """
    img = bg.copy()
    truth = []
    for (kind, (fx_, fy_), r), col in zip(SHAPES, FILLS):
        if kind in omit:
            continue
        # anchor: chosen in pixels, converted to the inches it stands for
        Xa, Ya, _ = cam.backproject(fx_ * w, fy_ * h, Z)
        mask = np.zeros((h, w), np.uint8)
        if kind == "circle":
            u, v = cam.project(Xa, Ya, Z)
            ax, ay = cam.pixel_radius(r, Z)
            cv2.ellipse(mask, (int(round(u)), int(round(v))),
                        (int(round(ax)), int(round(ay))), 0, 0, 360, 255, -1)
        else:
            pts = metric_points(kind, Xa, Ya, r)
            px = np.array([cam.project(X, Y, Z) for X, Y in pts], np.float32)
            cv2.fillPoly(mask, [np.round(px).astype(np.int32)], 255)

        layer = np.zeros_like(img)
        if gradient:
            other = (col[2], col[0], col[1])
            t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
            ramp = (np.float32(col) * (1 - t) + np.float32(other) * t)
            layer = np.repeat(ramp, h, axis=0).astype(np.uint8)
        else:
            layer[:] = col
        img[mask > 0] = layer[mask > 0]

        m = cv2.moments(mask)
        if m["m00"] == 0:
            continue
        truth.append((kind, cam.backproject(m["m10"] / m["m00"],
                                            m["m01"] / m["m00"], Z)))
    return img, truth


def match(dets, truth, cam, Z_true, tol_px=14.0):
    """Pair detections to truth, and separate the two ways a scene can fail.

    A shape can be missed by the *detector*, or found and then not placed in 3D
    because the frame held no usable ruler. They are different faults with
    different owners, so pairing is done in pixels -- which needs no scale -- and
    the 3D error is reported only for the shapes that got a position.
    """
    out, used = [], set()
    for kind, (Xt, Yt, Zt) in truth:
        ut, vt = cam.project(Xt, Yt, Z_true)
        best, bd = None, 1e18
        for i, d in enumerate(dets):
            if i in used:
                continue
            dd = np.hypot(d["center"][0] - ut, d["center"][1] - vt)
            if dd < bd:
                best, bd = i, dd
        if best is None or bd > tol_px:
            out.append((kind, None))
            continue
        used.add(best)
        d = dets[best]
        rec = {"px_err": bd, "named": d["shape"] == kind, "xyz": None}
        if d.get("xyz"):
            rec["xyz"] = d["xyz"]
            rec["xy_err"] = float(np.hypot(d["xyz"][0] - Xt, d["xyz"][1] - Yt))
            rec["z_err"] = d["xyz"][2] - Zt
        out.append((kind, rec))
    return out


# --------------------------------------------------------------------------
# 1 + 2. depth and position, over distance / background / fill
# --------------------------------------------------------------------------

def test_accuracy(w=1920, h=1080, save=None):
    print("1+2. depth and position vs ground truth\n")
    print(f"{'background':<12} {'fill':<9} {'true Z':>8} {'got Z':>8} {'Z err':>8} "
          f"{'Z err %':>8} {'XY err':>8} {'found':>7} {'placed':>7}")
    print("-" * 86)

    tiles, all_z, all_xy = [], [], []
    found = placed = expect = 0
    no_ruler = []
    for name, bg in backgrounds(w, h).items():
        for Z_true, grad in ((150.0, False), (250.0, False), (250.0, True),
                             (400.0, False)):
            cam = pose3d.Camera().for_frame((h, w))
            img, truth = render(cam, Z_true, w, h, bg, grad)
            dets, _, _ = da.detect(img)
            Z, src, _ = pose3d.solve_frame(dets, cam)
            rows = match(dets, truth, cam, Z_true)

            hit = [r for _, r in rows if r]
            loc = [r for r in hit if r["xyz"]]
            expect += len(rows); found += len(hit); placed += len(loc)
            fill = "gradient" if grad else "flat"
            if Z is None:
                no_ruler.append(f"{name}/{fill}@{Z_true:.0f}")
            xy = np.mean([r["xy_err"] for r in loc]) if loc else float("nan")
            zerr = (Z - Z_true) if Z else float("nan")
            if loc:
                all_xy.append(xy); all_z.append(abs(zerr))
            print(f"{name:<12} {fill:<9} {Z_true:>8.1f} "
                  f"{(Z if Z else float('nan')):>8.1f} {zerr:>+8.2f} "
                  f"{100 * zerr / Z_true:>+7.2f}% {xy:>7.2f}\" "
                  f"{len(hit)}/{len(rows):<5} {len(loc)}/{len(rows):<5}")

            if save and not grad and Z_true == 250.0:
                vis = img.copy()
                for d in dets:
                    cv2.drawContours(vis, [d["contour"]], -1, (0, 255, 255), 2)
                    if d.get("xyz"):
                        X, Y, Zc = d["xyz"]
                        cv2.putText(vis, f"{Zc:.0f}in", d["center"],
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                t = cv2.resize(vis, (480, 270))
                cv2.putText(t, f"{name} @ {Z_true:.0f}in", (8, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                tiles.append(t)

    print("-" * 86)
    print(f"detected {found}/{expect} ({100 * found / expect:.1f}%)   "
          f"placed in 3D {placed}/{found} of those   "
          f"mean |Z error| {np.mean(all_z):.2f} in   "
          f"mean XY error {np.mean(all_xy):.2f} in")
    if no_ruler:
        print(f"no usable ruler in {len(no_ruler)} scene(s): {', '.join(no_ruler)}")
        print("  -- the circle was segmented too raggedly to be trusted as one.")
        print("     A single frame has no other option; the video does, and takes it.")

    if save and tiles:
        while len(tiles) % 3:
            tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
        cv2.imwrite(save, grid)
        print(f"wrote {save}")
    return {"recall": found / expect, "placed": placed / max(found, 1),
            "z_err": float(np.mean(all_z)), "xy_err": float(np.mean(all_xy)),
            "no_ruler": no_ruler}


# --------------------------------------------------------------------------
# 3. the bootstrap: does the scale survive the circle leaving
# --------------------------------------------------------------------------

def test_bootstrap(w=1920, h=1080, Z_true=300.0, warmup=8):
    print("\n3. carrying the scale after the circle leaves\n")
    cam = pose3d.Camera().for_frame((h, w))
    bg = backgrounds(w, h)["asphalt"]
    plane = pose3d.PlaneScale(cam)

    print(f"{'frame':>6} {'circle':>8} {'source':>9} {'got Z':>8} {'err':>8}")
    print("-" * 44)
    ok = True
    for i in range(warmup + 4):
        seen = i < warmup
        img, truth = render(cam, Z_true, w, h, bg, omit=() if seen else ("circle",))
        dets, _, _ = da.detect(img)
        # no tracker here, so identity comes from matching against truth --
        # exactly the job the tracker does in the real pipeline.
        keys = []
        for d in dets:
            best, bd = "?", 1e18
            for kind, (Xt, Yt, _) in truth:
                dd = np.hypot(d["center"][0] - cam.project(Xt, Yt, Z_true)[0],
                              d["center"][1] - cam.project(Xt, Yt, Z_true)[1])
                if dd < bd:
                    best, bd = kind, dd
            keys.append(best)
        Z, src = plane.update([(k, d["shape"], d["area"], pose3d.is_measurable(d),
                                pose3d.circle_score(d["contour"]))
                               for k, d in zip(keys, dets)])
        err = (Z - Z_true) if Z else float("nan")
        if i >= warmup:
            ok &= (src == "learned" and abs(err) < 0.02 * Z_true)
        print(f"{i:>6} {'yes' if seen else 'NO':>8} {str(src):>9} "
              f"{(Z if Z else float('nan')):>8.2f} {err:>+8.2f}")
    print("-" * 44)
    print("PASS -- a shape the circle measured kept the scale" if ok
          else "FAIL -- scale lost when the circle left")
    return ok


# --------------------------------------------------------------------------
# 4. resolution invariance
# --------------------------------------------------------------------------

def test_resolution(Z_true=250.0):
    print("\n4. same scene, two resolutions\n")
    got = {}
    for w, h in ((1920, 1080), (960, 540)):
        cam = pose3d.Camera().for_frame((h, w))
        img, _ = render(cam, Z_true, w, h, backgrounds(w, h)["grass"])
        dets, _, _ = da.detect(img)
        Z, _, _ = pose3d.solve_frame(dets, cam)
        circ = next((d for d in dets if d["shape"] == "circle"), None)
        got[w] = (Z, circ["xyz"] if circ else None)
        print(f"  {w}x{h}   fx {cam.fx:8.2f}   Z {Z:8.2f} in   "
              f"circle X {circ['xyz'][0]:7.2f}  Y {circ['xyz'][1]:6.2f}")
    dz = abs(got[1920][0] - got[960][0])
    dx = abs(got[1920][1][0] - got[960][1][0])
    ok = dz < 0.02 * Z_true and dx < 2.0
    print(f"  difference: Z {dz:.2f} in, X {dx:.2f} in  ->  "
          + ("PASS" if ok else "FAIL"))
    return ok


# --------------------------------------------------------------------------
# 5. the arithmetic itself
# --------------------------------------------------------------------------

def test_roundtrip():
    print("\n5. project . backproject == identity\n")
    worst = 0.0
    for pp in ("given", "center"):
        cam = pose3d.Camera(principal=pp).for_frame((1080, 1920))
        for u, v, Z in [(0, 0, 100), (960, 540, 250), (1919, 1079, 600),
                        (37, 900, 1000)]:
            X, Y, Zc = cam.backproject(u, v, Z)
            u2, v2 = cam.project(X, Y, Zc)
            worst = max(worst, abs(u2 - u), abs(v2 - v))
    # depth must not depend on where the principal point is
    a = pose3d.Camera(principal="given").for_frame((1080, 1920))
    b = pose3d.Camera(principal="center").for_frame((1080, 1920))
    dz = abs(a.depth_from_area(32722, pose3d.CIRCLE_AREA_IN2)
             - b.depth_from_area(32722, pose3d.CIRCLE_AREA_IN2))
    ok = worst < 1e-9 and dz < 1e-9
    print(f"  worst pixel round-trip error {worst:.2e} px")
    print(f"  depth shift when the principal point moves {dz:.2e} in")
    print("  " + ("PASS" if ok else "FAIL"))
    return ok


# --------------------------------------------------------------------------
# 6. how far away this still works
# --------------------------------------------------------------------------

def test_envelope(w=1920, h=1080):
    """Sweep depth until detection fails, and report where.

    A drone changes altitude, so "how accurate is the depth" is only half the
    answer -- the other half is the range over which it can be measured at all.
    The limit is not in the arithmetic, which is exact at any distance, but in
    the detector: past a certain apparent size a shape stops being separable
    from the ground. Measuring where that happens is more useful than asserting
    it does not, so the number is reported rather than avoided.
    """
    print("\n6. operating envelope\n")
    print(f"{'Z (in)':>8} {'Z (ft)':>8} {'circle r':>10} {'found':>16} {'Z err':>9}")
    print("-" * 56)
    bgs = backgrounds(w, h)
    last_good = None
    for Z_true in (100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 600.0, 800.0):
        cam = pose3d.Camera().for_frame((h, w))
        if not scene_fits(cam, Z_true, w, h):
            print(f"{Z_true:>8.0f} {Z_true / 12:>8.1f} "
                  f"{cam.pixel_radius(pose3d.CIRCLE_RADIUS_IN, Z_true)[0]:>9.1f}px "
                  f"{'  scene clipped':>16} {'--':>9}")
            continue
        got, errs = 0, []
        for name, bg in bgs.items():
            img, truth = render(cam, Z_true, w, h, bg)
            dets, _, _ = da.detect(img)
            Z, _, _ = pose3d.solve_frame(dets, cam)
            got += sum(1 for _, r in match(dets, truth, cam, Z_true) if r)
            if Z:
                errs.append(abs(Z - Z_true) / Z_true)
        n = 5 * len(bgs)
        err = f"{100 * np.mean(errs):.2f}%" if errs else "--"
        print(f"{Z_true:>8.0f} {Z_true / 12:>8.1f} "
              f"{cam.pixel_radius(pose3d.CIRCLE_RADIUS_IN, Z_true)[0]:>9.1f}px "
              f"{got:>10}/{n:<5} {err:>9}")
        if got >= 0.9 * n:
            last_good = Z_true
    print("-" * 56)
    print(f"reliable out to about {last_good:.0f} in ({last_good / 12:.1f} ft), "
          f"where the circle is ~"
          f"{pose3d.Camera().for_frame((h, w)).pixel_radius(10, last_good)[0]:.0f} px "
          "in radius.")
    print("  Beyond that the detector -- not the camera model -- gives out: the")
    print("  texture window used to judge an interior is a fixed size, so once a")
    print("  shape is small enough that the window straddles its boundary, the")
    print("  interior no longer measures as smooth.")
    return last_good


# --------------------------------------------------------------------------
# 7. a real altitude change, versus a one-frame excursion
# --------------------------------------------------------------------------

def test_altitude(w=1920, h=1080, lo=250.0, hi=330.0, n=17):
    """The rate limit must reject noise without also rejecting the drone.

    PlaneScale clips the frame-to-frame change in depth, which is what stops a
    single bad segmentation from throwing the altitude by tens of per cent. A
    limit that also flattened a genuine climb would be worse than the problem it
    solves, so the climb is simulated and the recovered depth checked against it.
    Here the camera rises 80 in over 17 frames -- about 6 ft/s at 30 fps, brisker
    than the footage ever moves.
    """
    print("\n7. following a real altitude change\n")
    cam = pose3d.Camera().for_frame((h, w))
    bg = backgrounds(w, h)["grass"]
    plane = pose3d.PlaneScale(cam)

    print(f"{'frame':>6} {'true Z':>8} {'got Z':>8} {'err':>8} {'err %':>8}")
    print("-" * 42)
    errs = []
    for i, Z_true in enumerate(np.linspace(lo, hi, n)):
        img, truth = render(cam, Z_true, w, h, bg)
        dets, _, _ = da.detect(img)
        keys = []
        for d in dets:
            best, bd = "?", 1e18
            for kind, (Xt, Yt, _) in truth:
                ut, vt = cam.project(Xt, Yt, Z_true)
                dd = np.hypot(d["center"][0] - ut, d["center"][1] - vt)
                if dd < bd:
                    best, bd = kind, dd
            keys.append(best)
        Z, src = plane.update([(k, d["shape"], d["area"], pose3d.is_measurable(d),
                                pose3d.circle_score(d["contour"]))
                               for k, d in zip(keys, dets)])
        if Z:
            errs.append(abs(Z - Z_true) / Z_true)
        if i % 4 == 0 or i == n - 1:
            print(f"{i:>6} {Z_true:>8.1f} {(Z if Z else float('nan')):>8.1f} "
                  f"{(Z - Z_true if Z else float('nan')):>+8.2f} "
                  f"{(100 * (Z - Z_true) / Z_true if Z else float('nan')):>+7.2f}%")
    print("-" * 42)
    ok = bool(errs) and max(errs) < 0.02
    print(f"climbed {hi - lo:.0f} in; worst tracking error {100 * max(errs):.2f}%  ->  "
          + ("PASS" if ok else "FAIL"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default=None, help="write a contact sheet here")
    args = ap.parse_args()

    acc = test_accuracy(save=args.save)
    boot = test_bootstrap()
    res = test_resolution()
    rt = test_roundtrip()
    env = test_envelope()
    alt = test_altitude()

    # The 3D stage is judged on what it does with the shapes the detector hands
    # it. Detection recall is step 3's number and is reported, not re-graded here.
    ok = (acc["recall"] >= 0.90 and acc["z_err"] < 3.0 and acc["xy_err"] < 2.0
          and boot and res and rt and alt)
    print("\n" + "=" * 60)
    print(f"detected {100 * acc['recall']:.1f}%   placed {100 * acc['placed']:.1f}%   |Z| err {acc['z_err']:.2f} in   "
          f"XY err {acc['xy_err']:.2f} in   bootstrap {'ok' if boot else 'FAIL'}   "
          f"resolution {'ok' if res else 'FAIL'}   maths {'ok' if rt else 'FAIL'}   "
          f"altitude {'ok' if alt else 'FAIL'}")
    print(f"usable to {env:.0f} in ({env / 12:.1f} ft) at 1080p")
    print("3D STAGE: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
