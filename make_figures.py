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

print("wrote 6 figures to", OUT)
