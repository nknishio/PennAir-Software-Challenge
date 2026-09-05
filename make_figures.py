"""Regenerate every figure used in README.md. Run: python make_figures.py"""
import cv2, numpy as np, detect_shapes as ds, detect_shapes_agnostic as da

OUT = "figures"
F = cv2.FONT_HERSHEY_SIMPLEX


def label(img, text, y=30, color=(255, 255, 255)):
    cv2.putText(img, text, (12, y), F, 0.75, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), F, 0.75, color, 2, cv2.LINE_AA)
    return img


def strip(panels, w=420):
    out = []
    for img, cap in panels:
        h = int(img.shape[0] * w / img.shape[1])
        t = cv2.resize(img, (w, h))
        if t.ndim == 2:
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        out.append(label(t, cap))
    return np.hstack(out)


def norm(x):
    return cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def frame(path, n):
    c = cv2.VideoCapture(path)
    c.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, img = c.read()
    c.release()
    return img


# 1 -- why texture and not colour -------------------------------------------
img = cv2.imread("PennAir 2024 App Static.png")
hue = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]
hue_vis = cv2.applyColorMap(cv2.normalize(hue, None, 0, 255, cv2.NORM_MINMAX), cv2.COLORMAP_HSV)
_, _, energy = ds.detect(img)   # detect() returns (results, mask, texture_map)
cv2.imwrite(f"{OUT}/01_why_texture.png", strip([
    (img, "1. input"),
    (hue_vis, "2. hue - trapezoid == grass"),
    (norm(energy), "3. texture - shapes are black"),
]))

# 2 -- the static pipeline ---------------------------------------------------
mask, _ = ds.smoothness_mask(img)
res, _, _ = ds.detect(img)
cv2.imwrite(f"{OUT}/02_static_pipeline.png", strip([
    (img, "input"),
    (norm(energy), "stage 1: texture map"),
    (mask, "stage 1: seed mask"),
    (ds.annotate(img, res), "stages 2-4: result"),
]))

# 3 -- occlusion: one blob, two shapes ---------------------------------------
g600 = frame("PennAir 2024 App Dynamic.mp4", 600)
m600, _ = ds.smoothness_mask(g600)
r600, _, _ = ds.detect(g600)
box = (430, 560, 400, 460)
crop = lambda im: im[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
cv2.imwrite(f"{OUT}/03_occlusion.png", strip([
    (crop(g600), "circle over rectangle"),
    (crop(m600), "one merged blob"),
    (crop(ds.annotate(g600, r600)), "two shapes, two centres"),
], w=330))

# 4 -- video tracking --------------------------------------------------------
tiles = [cv2.resize(frame("output_dynamic.mp4", n), (640, 360)) for n in (240, 700, 1450)]
cv2.imwrite(f"{OUT}/04_video_tracking.png", np.hstack(tiles))

# 5 -- what the hard video breaks -------------------------------------------
h0 = frame("PennAir 2024 App Dynamic Hard.mp4", 0)
old_mask, _ = ds.smoothness_mask(h0)                       # variance measure
en = da.texture_energy(h0)                                 # high-pass residual
new_mask = (en < da.relative_threshold(en)).astype(np.uint8) * 255
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, k, iterations=2)
new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, k, iterations=3)
zoom = (250, 600, 1250, 420)                               # the two gradient shapes
z = lambda im: im[zoom[1]:zoom[1] + zoom[3], zoom[0]:zoom[0] + zoom[2]]
cv2.imwrite(f"{OUT}/05_gradient_problem.png", strip([
    (z(h0), "gradient fills"),
    (z(old_mask), "old: shapes half-lost"),
    (z(new_mask), "new: both recovered"),
], w=430))

# 6 -- hard video result -----------------------------------------------------
tiles = [cv2.resize(frame("output_hard.mp4", n), (640, 360)) for n in (60, 800, 1800)]
cv2.imwrite(f"{OUT}/06_hard_result.png", np.hstack(tiles))

# 7 -- 3D: what the camera model adds ---------------------------------------
# Two frames of the hard footage carrying metric coordinates, one measured from
# the circle and one from a shape the circle taught.
import detect_video_3d as dv3, pose3d, detect_video_agnostic as dva

cap = cv2.VideoCapture("PennAir 2024 App Dynamic Hard.mp4")
cam = None; tracker = None; plane = None; picked = []
for i in range(150):
    ok, f = cap.read()
    if not ok:
        break
    dets, _, _ = da.detect(f)
    if tracker is None:
        tracker = dva.ShapeTracker(f.shape)
        cam = pose3d.Camera().for_frame(f.shape)
        plane = pose3d.PlaneScale(cam)
    tracks = tracker.update(dets, i)
    Z, src = plane.update([(t.id, t.label, t.area, dv3.measurable(t, f.shape),
                            pose3d.circle_score(t.contour)) for t in tracks])
    for t in tracks:
        t.xyz = plane.locate(t.center, Z) if Z else None
    # one frame measured by the circle itself, one measured by a shape the
    # circle taught -- the whole point of the bootstrap, side by side
    want = "circle" if not picked else "learned"
    if i > 30 and src == want and len(tracks) >= 4:
        picked.append(cv2.resize(
            dv3.draw_overlay(f, tracks, i, 12.0, dets, Z, src, "in"), (760, 428)))
    if len(picked) == 2:
        break
cap.release()
cv2.imwrite(f"{OUT}/07_3d_result.png", np.hstack(picked))

# 8 -- 3D against known metric truth ----------------------------------------
import test_pose3d as tp
tp.test_accuracy(save=f"{OUT}/08_3d_truth.png")

# 9 -- how the watershed refinement works ------------------------------------
# The step that replaced colour thresholding in the agnostic detector: markers
# say what is certainly inside and certainly outside, and the image decides the
# rest. Shown on the navy-to-yellow pentagon, which has no single fill colour.
p = da.auto_params(h0.shape)
win = p["win"]
seeds, _, _ = da.candidates(h0, p["min_area"])
pent = min(seeds, key=lambda c: abs(cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] // 2 - 1366))

r_out = win * 3
pad = r_out + win
bx, by, bw, bh = cv2.boundingRect(pent)
x0, y0 = max(bx - pad, 0), max(by - pad, 0)
x1, y1 = min(bx + bw + pad, h0.shape[1]), min(by + bh + pad, h0.shape[0])
roi = np.ascontiguousarray(h0[y0:y1, x0:x1])

seedm = np.zeros(roi.shape[:2], np.uint8)
cv2.drawContours(seedm, [pent - (x0, y0)], -1, 255, cv2.FILLED)
fg = cv2.erode(seedm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(win // 2, 3),) * 2))
away = cv2.distanceTransform(255 - seedm, cv2.DIST_L2, 3)
mk = np.zeros(roi.shape[:2], np.int32)
mk[away > r_out] = 1                       # certainly background
mk[fg > 0] = 2                             # certainly shape

zones = roi.copy()
for sel, tint, a in ((mk == 1, (40, 40, 220), 0.55),      # red   = outside
                     (mk == 2, (60, 220, 60), 0.55),      # green = inside
                     (mk == 0, (255, 255, 255), 0.45)):   # white = undecided
    zones[sel] = ((1 - a) * zones[sel] + a * np.float32(tint)).astype(np.uint8)

seedvis = roi.copy()
cv2.drawContours(seedvis, [pent - (x0, y0)], -1, (0, 0, 255), 2)
mk_base = mk.copy()                        # watershed overwrites its markers
cv2.watershed(roi, mk)
cnts, _ = cv2.findContours((mk == 2).astype(np.uint8) * 255,
                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
fin_contour = max(cnts, key=cv2.contourArea)
final = roi.copy()
cv2.drawContours(final, [fin_contour], -1, (0, 255, 255), 3)

cv2.imwrite(f"{OUT}/09_watershed.png", strip([
    (seedvis, "1. rough seed (red)"),
    (zones, "2. green=in  red=out  white=undecided"),
    (final, "3. watershed's answer"),
], w=330))

# 10 -- why a rounded seed yields a sharp answer -----------------------------
# The marker's outline never reaches the result. What shapes the answer is the
# gradient "terrain" the flood runs over, and that terrain is razor sharp.
terrain = boundary_energy_vis = da.boundary_energy(roi)
tv = cv2.applyColorMap(
    cv2.normalize(np.clip(terrain, 0, 120), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
    cv2.COLORMAP_INFERNO)

start_cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
start = roi.copy()
cv2.drawContours(start, start_cnts, -1, (60, 255, 60), 2)

both = roi.copy()
cv2.drawContours(both, start_cnts, -1, (60, 255, 60), 2)   # where the water began
cv2.drawContours(both, [fin_contour], -1, (0, 255, 255), 2)  # where it stopped

cv2.imwrite(f"{OUT}/10_flood.png", strip([
    (start, "1. where water starts (green)"),
    (tv, "2. the terrain: bright = ridge"),
    (both, "3. green -> yellow (where it stopped)"),
], w=330))

# 11 -- the background-agnosticism suite -------------------------------------
# Nine synthetic backgrounds, all with *gradient* fills -- the harder case, so
# passing here implies the flat case. Ground truth is exact: the scene is
# generated, and the expected centre is measured from each shape's own mask.
import test_backgrounds as tb

bg_tiles = []
for bg_name, bg_img in tb.backgrounds().items():
    scene = tb.draw_shapes(bg_img, True)
    score = tb.evaluate(scene)
    vis = scene.copy()
    for d in da.detect(scene)[0]:
        cv2.drawContours(vis, [d["contour"]], -1, (0, 255, 255), 3)
        cv2.drawMarker(vis, d["center"], (0, 0, 255), cv2.MARKER_CROSS, 26, 3)
        org = (d["center"][0] - 52, d["center"][1] - 46)
        cv2.putText(vis, d["shape"], org, F, 0.62, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(vis, d["shape"], org, F, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    t = cv2.resize(vis, (430, 242))
    cv2.rectangle(t, (0, 0), (430, 26), (0, 0, 0), -1)
    cv2.putText(t, f"{bg_name}  {score['found']}/5", (7, 19), F, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    bg_tiles.append(cv2.copyMakeBorder(t, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40)))

cv2.imwrite(f"{OUT}/11_background_suite.png",
            np.vstack([np.hstack(bg_tiles[i:i + 3]) for i in range(0, 9, 3)]))

# 12 -- poster frames for the README clips ------------------------------------
# The clips are committed; the full-length outputs are gitignored. So these read
# from the clips, and a fresh checkout can regenerate them.
CLIPS = [
    ("output_dynamic_CLIP.mp4", 155, "clip_02_video.png"),
    ("output_hard_CLIP.mp4", 130, "clip_03_agnostic.png"),
    ("output_hard_3d_CLIP.mp4", 130, "clip_04_3d.png"),
]
posters = 0
for src, frame_no, name in CLIPS:
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, f = cap.read()
    cap.release()
    if not ok:
        print(f"  skipped {name}: could not read {src}")
        continue
    w = 900
    h = int(f.shape[0] * w / f.shape[1])
    img = (cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) * 0.82).astype(np.uint8)
    cx, cy, r = w // 2, h // 2, int(h * 0.11)
    halo = img.copy()
    cv2.circle(halo, (cx, cy), r, (255, 255, 255), -1)
    img = cv2.addWeighted(halo, 0.30, img, 0.70, 0)
    cv2.circle(img, (cx, cy), r, (255, 255, 255), max(2, r // 18), cv2.LINE_AA)
    t = int(r * 0.52)
    cv2.fillPoly(img, [np.array([[cx - t // 2, cy - t], [cx - t // 2, cy + t],
                                 [cx + int(t * 0.95), cy]], np.int32)],
                 (255, 255, 255), cv2.LINE_AA)
    cv2.imwrite(f"{OUT}/{name}", img)
    posters += 1

print(f"wrote 11 figures + {posters} clip posters to", OUT)
