# Part 5 demo script

A terminal-only walkthrough of the ROS 2 layer, verified working on
2026-09-08. Roughly 6 minutes as written, with optional extensions at the end.

**Environment note.** The demo runs in Docker (`pennair:ros2`), which has no GUI — no RViz, no
`rqt_image_view`. That is fine: the ROS graph, the interfaces, the payload and the QoS are all
inspectable from the terminal, and the visual proof is the pre-rendered clip in the README. If a
GUI is required, use the Ubuntu VM route in [`ROS2_SETUP.md`](ROS2_SETUP.md) instead.

---

## Pre-flight (10 minutes before, not during)

Docker Desktop must be running. The `docker` CLI is **not** on the default PATH:

```bash
export PATH="$HOME/.docker/bin:$PATH"
docker --version                      # confirm before you rely on it
```

Start a **named** container from the saved snapshot. The snapshot already contains
`cv_bridge`, `vision_msgs`, OpenCV and colcon, so there is nothing to install:

```bash
cd "/Users/nelson/Documents/GitHub Projects/PennAir Challenge"
docker run -it --name pennair -v "$PWD":/work -w /work pennair:ros2 bash
```

> **Do not use `--rm`.** The original container was started that way, which is why the
> environment was one closed shell away from being destroyed. `pennair:ros2` is a commit of
> that container so it cannot happen again. Reuse it with `docker start -ai pennair`.

Inside the container, source and confirm the build:

```bash
source /opt/ros/jazzy/setup.bash
source /work/ros2_ws/install/setup.bash
ros2 pkg list | grep pennair          # expect pennair_msgs, pennair_vision
```

If `install/` is missing or stale:

```bash
cd /work/ros2_ws && colcon build --symlink-install && source install/setup.bash
```

Takes about 6 seconds. Then **do one full dry run** end to end. Never demo a cold path.

---

## Terminal layout

Two shells into the same container. Open the second with:

```bash
export PATH="$HOME/.docker/bin:$PATH"
docker exec -it pennair bash
```

| Terminal | Role |
|---|---|
| **A** | runs the launch file; shows live log output |
| **B** | runs the `ros2` inspection commands |

Source both:

```bash
source /opt/ros/jazzy/setup.bash && source /work/ros2_ws/install/setup.bash
```

> **The one thing that will ruin the demo:** two launches running at once. Two
> `video_publisher` nodes stream the same video from different positions, the detector
> interleaves both, and the tracked-shape count doubles. ROS 2 warns about it
> (`nodes in the graph that share an exact name`). Before launching, always check terminal A is
> idle, or run `ros2 node list` and confirm nothing is up.

---

## The demo

### 1. Frame the problem (30 s, no commands)

> Parts 1–4 produce a detector that finds shapes, tracks them, and reports each centre in metres
> using the camera intrinsics. Part 5 makes that available to the rest of a robot: one node
> streams frames, another runs detection and publishes positions and outlines on topics.
>
> The port is small because the detector was already written for it — one frame at a time,
> detection a pure function of that frame, all state in the tracker. A subscriber callback is
> the same shape, so the detector node's body is six lines lifted from the CLI version.

### 2. Custom interfaces (45 s)

Lead with the message design — it shows deliberate schema work rather than reusing a stock type.

```bash
ros2 interface show pennair_msgs/msg/ShapeDetectionArray
```

Point at three fields:

- `plane_depth` — one depth per **frame**, not per shape, because the flat-surface assumption
  means a single depth describes the whole scene.
- `depth_source` — `circle` when the 10 in reference circle was measured directly, `learned`
  when a shape the circle previously sized carried the scale. The consumer can tell how the
  number was obtained.
- `outline` — the refined contour, not a bounding box and not a polygon approximation.

Mention the packaging decision: **interfaces live in a separate `ament_cmake` package** because
message generation runs through `rosidl`, which a pure-Python package cannot invoke.

### 3. Start the system (30 s)

**Terminal A:**

```bash
ros2 launch pennair_vision shapes.launch.py \
  video:="/work/PennAir 2024 App Dynamic Hard.mp4" \
  scale:=1.0 rate:=5.0 rviz:=false
```

`scale:=1.0` matters — see [Settings that matter](#settings-that-matter). Wait for the log to
reach roughly frame 50, then read a line aloud:

```
[shape_detector] frame 100  97.1 ms  10.3 fps  tracking 5  depth 6.40 m [circle]
```

Five shapes tracked, depth 6.4 m, measured from the circle.

### 4. The graph is real (45 s)

**Terminal B:**

```bash
ros2 node list
ros2 topic list
```

Three nodes, nine topics. Then explain the pairing that carries the calibration:

```bash
ros2 topic echo --once /camera/camera_info | head -12
```

> The publisher sends **K** on a topic rather than the detector hardcoding it. That is idiomatic
> ROS, and it removes a specific way to be silently wrong: `scale` resizes frames, intrinsics
> are measured in pixels, so a resized image needs a resized K. If it didn't scale, half-scale
> streaming would report every shape at twice its true distance and nothing would look broken.

### 5. The payload (60 s)

```bash
ros2 topic echo --once /shapes/detections
```

It is long, so scroll to the top and read out: `plane_depth`, `depth_source`, then one detection
with its `shape`, `track_id`, `position` and the length of `outline`.

Worth saying: **positions are in metres** per REP-103, while the algorithm works internally in
inches — conversion happens only at the publish boundary. And `frame_id` is
`camera_optical_frame`, whose x-right / y-down / z-forward convention is exactly what a pinhole
model produces, so no axis remapping is needed anywhere.

### 6. The money moment (30 s)

This is the strongest single claim in the demo.

> The command-line pipeline reports a plane depth of **251.74 inches** on this footage. That is
> **6.394 metres**. The ROS topic reports **6.3949**. The port did not change the answer, and if
> it ever disagreed by more than a fraction of a percent I would look at the K scaling or the
> unit conversion in the ROS layer, not at the detector.

Verify live if asked:

```bash
ros2 topic echo --once --field plane_depth /shapes/detections
```

### 7. A design decision under the hood (60 s)

```bash
ros2 topic info -v /camera/image_raw
```

Expected, and the point of showing it:

```
QoS profile:
  Reliability: BEST_EFFORT
  Durability: VOLATILE
```

> Both ends use the sensor-data QoS profile — best-effort, keep-last, depth 5. The detector
> runs around 10 fps against a
> publisher that does not wait for it, so it always works on the newest frame and discards the
> backlog. Frame dropping here is intended, not a shortfall — a reliable, deep queue would
> accumulate unbounded lag and report positions for a scene that had already moved on. It also
> means this is the first time the streaming contract from Part 2 is genuinely exercised:
> frames really do arrive asynchronously.

This command also shows **publisher and subscription counts**, which is how you catch the
duplicate-node mistake warned about above.

### 8. Parameters (30 s, optional)

```bash
ros2 param list /shape_detector
ros2 param get /shape_detector principal_point
```

> Everything configurable is a ROS parameter with a launch argument in front of it —
> `principal_point` selects whether the supplied K is used verbatim (`cx = cy = 0`, the optical
> axis at the top-left pixel) or recentred. Depth is identical either way, since it depends only
> on `fx` and `fy`.

---

## Settings that matter

| Argument | Use | Why |
|---|---|---|
| `scale:=1.0` | **always, for a demo** | at the `0.5` default the shapes shrink into the range where classification degrades: measured labels went `circle → pentagon` and `trapezoid → hexagon`, and depth drifted from 0.01% to 0.74% off |
| `rate:=5.0` | demo | the detector needs ~97 ms/frame at full resolution; 5 Hz keeps the log readable |
| `rviz:=false` | in Docker | there is no display; leaving it on prints a harmless failure |
| `principal_point:=center` | only with RViz | markers otherwise sit in one corner |

At `scale:=1.0` the expected labels are pentagon, circle, rectangle, triangle, trapezoid — all
five correct. The trapezoid's vote confidence sits around 0.42, which is the weakness documented
in the README rather than a surprise.

---

## If it breaks

| Symptom | Cause | Fix |
|---|---|---|
| `docker: command not found` | CLI not on PATH | `export PATH="$HOME/.docker/bin:$PATH"` |
| `package 'pennair_vision' not found` | workspace not sourced in this shell | `source /work/ros2_ws/install/setup.bash` |
| Tracks double (8–10 instead of 5) | two launches running | Ctrl-C terminal A; `ros2 node list` to confirm; relaunch |
| `nodes ... share an exact name` warning | same | same |
| `could not open /work/...mp4` | path not quoted — the filename has spaces | `video:="/work/PennAir 2024 App Dynamic Hard.mp4"` |
| `ModuleNotFoundError: pose3d` | nodes cannot find the algorithm at the repo root | `export PENNAIR_ROOT=/work`, or pass `repo_root:=/work` |
| `/shapes/detections` silent | no frames arriving | `ros2 topic hz /camera/image_raw` — but see the note below |
| `plane_depth` is 0.0 | circle not yet seen | normal for the first frames; if it persists, the circle is not being detected |

> **Do not trust `ros2 topic hz` on `/camera/image_raw`.** It is a best-effort topic, so
> the tool drops samples and under-reports — it showed 7.5 Hz for a 10 Hz publisher. Use the
> detector's own log line for the real rate.

**If the live demo fails entirely**, fall back to the recorded evidence and keep talking: the
annotated clip in [`README.md`](README.md) Part 4, and the topic table in Part 5. Then say what
you would check first, which is a better answer than a working demo you cannot explain.

---

## Optional extensions

**Record a bag** — shows you know the standard way to capture and replay robot data:

```bash
ros2 bag record /camera/image_raw /shapes/detections /shapes/markers -o /work/pennair_demo
ros2 bag info /work/pennair_demo
```

**Demonstrate a QoS mismatch** — the strongest live-debug story available, because ROS 2 names
the offending policy for you. Ask for a reliable subscription to a best-effort publisher:

```bash
ros2 topic echo --once --qos-reliability reliable --field header.frame_id /camera/image_raw
```

```
[WARN] [_ros2cli]: New publisher discovered on topic '/camera/image_raw', offering
incompatible QoS. No messages will be received from it.
Last incompatible policy: RELIABILITY
```

Nothing arrives, and the warning says exactly why. Switching to `best_effort` makes it flow:

```bash
ros2 topic echo --once --qos-reliability best_effort --field header.frame_id /camera/image_raw
```

That one may also print `A message was lost!!!`, which is the echo tool noticing a dropped
sample — best-effort delivery working as specified, not an error.

> This is the canonical "why aren't my two nodes talking" bug. Worth knowing that a
> `RELIABLE` subscriber and a `BEST_EFFORT` publisher are **incompatible and never connect**,
> and that `ros2 topic info -v` plus that warning are how you find it.

**Show the algorithm is untouched** — the point that the ROS layer is a layer, not a rewrite:

```bash
cd /work && python3 run_tests.py --step 4
```

---

## Questions to expect

| Question | Short answer |
|---|---|
| Why two packages? | Message generation needs `rosidl` under `ament_cmake`; an `ament_python` package cannot run it |
| Why a custom message *and* `vision_msgs`? | Fidelity versus interoperability — the outline, track ID and depth provenance have nowhere to live in a stock type |
| Why publish `CameraInfo`? | Idiomatic, and it makes the intrinsics scale with the image so `scale` cannot silently corrupt depth |
| Why best-effort QoS? | The detector is slower than the source; newest-frame-wins is correct for a live feed |
| Where does the algorithm live? | Repository root, imported via `PENNAIR_ROOT`, so `run_tests.py` keeps working and there is no second copy |
| Services or actions? | Neither fits — this is a continuous sensor stream, which is what topics are for. A service would suit a one-shot "detect in this image" request |
| What next? | Lifecycle nodes for deterministic startup, a composable component for zero-copy intra-process transport, `image_transport` for compressed streaming, and unit tests on the message-building functions |
| Is it tested? | `test_ros_pipeline.py` asserts messages arrive, five detections are present, and `plane_depth` is within 5% of the CLI's 6.394 m |

---

## Cheat card

```bash
# host
export PATH="$HOME/.docker/bin:$PATH"
docker start -ai pennair            # or: docker run -it --name pennair -v "$PWD":/work -w /work pennair:ros2 bash
docker exec -it pennair bash        # second terminal

# in every shell
source /opt/ros/jazzy/setup.bash && source /work/ros2_ws/install/setup.bash

# A: run
ros2 launch pennair_vision shapes.launch.py \
  video:="/work/PennAir 2024 App Dynamic Hard.mp4" scale:=1.0 rate:=5.0 rviz:=false

# B: inspect
ros2 interface show pennair_msgs/msg/ShapeDetectionArray
ros2 node list
ros2 topic list
ros2 topic echo --once /camera/camera_info
ros2 topic echo --once /shapes/detections
ros2 topic echo --once --field plane_depth /shapes/detections    # expect ~6.394
ros2 topic info -v /camera/image_raw                             # BEST_EFFORT
ros2 param list /shape_detector
```
