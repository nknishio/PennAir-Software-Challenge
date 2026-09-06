# Getting the ROS 2 system running, from a blank Mac

Start to finish: a virtual machine, ROS 2, this repository, and the shape detector publishing
on topics. Roughly **45–60 minutes**, most of it waiting on downloads.

Everything ROS lives in a Linux VM. Nothing is installed on macOS, and nothing on the Mac side
changes.

> **A correction to an earlier note.** UTM offers two virtualisation backends, and the important
> choice is **Virtualize** rather than **Emulate** — emulation translates ARM↔x86 and is
> painfully slow. Both of UTM's *virtualize* backends (QEMU and Apple Virtualization) run at
> near-native speed on Apple Silicon when the guest is ARM64, so this is not the QEMU-is-slow
> trade-off it sounds like. For a Linux **desktop** guest, use UTM's default **QEMU** backend:
> it handles display resizing, clipboard sharing and networking more reliably than Apple
> Virtualization does for Linux. Leave the "Use Apple Virtualization" box unticked.

---

## Step 1 — Download the Ubuntu ISO

You need an **ARM64** (also written `aarch64`) image. An x86 ISO will not boot without emulation.

Go to <https://cdimage.ubuntu.com/releases/24.04/release/> and take, in order of preference:

| File | Then |
|---|---|
| `ubuntu-24.04.x-desktop-arm64.iso` | nothing extra — a desktop is included |
| `ubuntu-24.04.x-live-server-arm64.iso` | add a desktop in Step 4 |

Either works. The server ISO is smaller and definitely present; the desktop ISO saves you one
command later. **You do need a graphical desktop eventually** — RViz and `rqt_image_view` are
GUI applications.

---

## Step 2 — Create the VM in UTM

1. Open UTM → **Create a New Virtual Machine**.
2. Choose **Virtualize**. *(Not Emulate.)*
3. Choose **Linux**.
4. Leave **"Use Apple Virtualization" unticked** — see the note above.
5. **Boot ISO Image** → *Browse* → select the `.iso` you downloaded.
6. Hardware:

   | Setting | Value | Why |
   |---|---|---|
   | Memory | **8192 MB** | ROS desktop plus a build; 4 GB will thrash |
   | CPU Cores | **6** | leaves headroom on an M3 Max |
   | Storage | **64 GB** | thin-provisioned, so it only uses what it writes |

7. **Shared Directory**: skip it. Files go in over SSH in Step 7, which is more reliable than
   UTM's shared-folder mechanism.
8. Name it `ros2-jazzy`, **Save**, then press ▶ to boot.

---

## Step 3 — Install Ubuntu

Follow the installer. The only choices that matter:

- **Minimal installation** is fine and faster.
- **Erase disk and install Ubuntu** — this is the VM's virtual disk, not your Mac.
- Set a username and password you will type often. This guide assumes `nelson`.
- Tick **"Install third-party software"** if offered.

When it says installation is complete, **shut the VM down** (don't just reboot). In UTM, open the
VM's settings, remove the ISO from the CD/DVD drive, then start it again — otherwise it boots
back into the installer.

---

## Step 4 — First boot

Open a terminal in the VM (Ctrl+Alt+T) and run:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y openssh-server curl git
```

**Only if you used the server ISO**, add a desktop now:

```bash
sudo apt install -y ubuntu-desktop-minimal
sudo reboot
```

Then find the VM's IP address and note it down — you need it in Step 7:

```bash
ip -4 addr show | grep -oP '(?<=inet\s)192\.168\.\d+\.\d+'
```

Check the Mac can reach it. **On the Mac**, in a normal terminal:

```bash
ssh nelson@<VM_IP>
```

If that logs in, the rest is easy. If it refuses, confirm UTM's network mode is **Shared
Network** (the default) in the VM's settings.

---

## Step 5 — Install ROS 2 Jazzy

```bash
sudo add-apt-repository universe -y
sudo apt update && sudo apt install -y curl

export ROS_APT_SOURCE_VERSION=$(curl -s \
  https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
  | grep -F "tag_name" | awk -F\" '{print $4}')

curl -L -o /tmp/ros2-apt-source.deb \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $UBUNTU_CODENAME)_all.deb"

sudo apt install -y /tmp/ros2-apt-source.deb
sudo apt update
sudo apt install -y ros-jazzy-desktop
```

> The ROS apt repository moved to this signed-package arrangement after a GPG key rotation. If
> any URL above 404s, the commands have drifted — follow
> <https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html> verbatim instead. That
> page is the authority; everything else here is unaffected.

Make it available in every new shell, and prove it works:

```bash
echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc
source ~/.bashrc
ros2 topic list          # expect /parameter_events and /rosout
```

---

## Step 6 — Project dependencies

```bash
sudo apt install -y \
    ros-jazzy-cv-bridge \
    ros-jazzy-vision-msgs \
    ros-jazzy-rqt-image-view \
    python3-opencv \
    python3-numpy \
    python3-colcon-common-extensions
```

> ### The one mistake that costs an afternoon
>
> **Do not `pip install opencv-python` or `numpy` in this VM.**
>
> `cv_bridge` is a compiled library built against the *system* NumPy and OpenCV. A pip-installed
> NumPy 2.x sits in front of the system one on the import path, and `cv_bridge` then fails with
> an ABI error that names neither pip nor NumPy and reads like a totally unrelated problem.
>
> Ubuntu 24.04 will refuse the pip install anyway (PEP 668) and suggest
> `--break-system-packages`. **Do not use that flag here.** It does exactly what it says.
>
> The detector needs nothing beyond OpenCV and NumPy — every module in this repository imports
> only `cv2`, `numpy` and the standard library — so the apt packages above are sufficient.

---

## Step 7 — Get the repository and the videos into the VM

The two source videos are in `.gitignore` (97 MB and 50 MB), so they never come from a clone
regardless of which route you take.

**Route A — copy everything from the Mac (works right now).** On the **Mac**:

```bash
cd "/Users/nelson/Documents/GitHub Projects/PennAir Challenge"

rsync -av --progress \
  --exclude 'output_*.mp4' --exclude '.git' --exclude '__pycache__' \
  --exclude 'ros2_ws/build' --exclude 'ros2_ws/install' --exclude 'ros2_ws/log' \
  ./ nelson@<VM_IP>:~/pennair/
```

About 160 MB, mostly the two videos. This is the route to use if the ROS work has not been
pushed to GitHub yet.

**Route B — clone, then copy only the videos.** Only correct once `ros2_ws/`, `pose3d.py`,
`detect_3d.py`, `detect_video_3d.py`, `run_tests.py` and `test_pose3d.py` are committed and
pushed. In the **VM**:

```bash
git clone https://github.com/nknishio/PennAir-Software-Challenge.git ~/pennair
```

then from the **Mac**:

```bash
cd "/Users/nelson/Documents/GitHub Projects/PennAir Challenge"
scp "PennAir 2024 App Dynamic.mp4" "PennAir 2024 App Dynamic Hard.mp4" \
    nelson@<VM_IP>:~/pennair/
```

---

## Step 8 — Check the algorithm before touching ROS

This matters. It separates "my VM is misconfigured" from "my ROS package is wrong", and those
two failures look identical once they are tangled together.

In the **VM**:

```bash
cd ~/pennair
python3 run_tests.py
```

Expect `4/4 steps passed`, in a couple of minutes. If this fails, **stop and fix it here** —
ROS cannot help and will only add noise. The likely cause is a missing OpenCV (`python3-opencv`)
or a missing video file.

---

## Step 9 — Build the ROS workspace

```bash
cd ~/pennair/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

`pennair_msgs` builds first (it generates the message types), then `pennair_vision`. Confirm the
interfaces exist:

```bash
ros2 interface show pennair_msgs/msg/ShapeDetection
```

> `source install/setup.bash` applies to the current terminal only. Every new terminal needs it
> again. Adding it to `~/.bashrc` is convenient but hides errors when the workspace is broken.

---

## Step 10 — Run it

```bash
cd ~/pennair/ros2_ws
source install/setup.bash

ros2 launch pennair_vision shapes.launch.py \
    video:=$HOME/pennair/"PennAir 2024 App Dynamic Hard.mp4" \
    principal_point:=center \
    rviz:=true
```

`principal_point:=center` is a display choice. The supplied K has `cx = cy = 0`, which puts the
optical axis at the top-left pixel, so with `given` every marker sits off in one corner of RViz.
Depth is identical either way — see the principal-point note in [README.md](README.md).

---

## Step 11 — Verify

In a **second terminal** (`source install/setup.bash` first):

```bash
ros2 topic list
ros2 topic hz /camera/image_raw          # ~10 Hz
ros2 topic hz /shapes/detections         # lower — the detector's real rate
ros2 topic echo /shapes/detections --once
```

**The number that proves the whole chain.** `plane_depth` should read about **6.39** (metres).
The command-line pipeline reports 251.74 inches on this footage, which is the same distance. If
it disagrees by more than a few percent, the fault is in the ROS layer's camera scaling or unit
conversion — not in the detector.

Watch it:

```bash
ros2 run rqt_image_view rqt_image_view      # choose /shapes/image_annotated
```

And the automated check:

```bash
cd ~/pennair/ros2_ws
PENNAIR_VIDEO=$HOME/pennair/"PennAir 2024 App Dynamic Hard.mp4" \
  python3 -m pytest src/pennair_vision/test/test_ros_pipeline.py -s
```

Record something reproducible for the writeup:

```bash
ros2 bag record /camera/image_raw /shapes/detections /shapes/markers -o pennair_demo
```

---

## What "working" looks like

The detector is slower than the video. That is expected and correct — the QoS is configured so
the node always takes the newest frame and drops the backlog, which is what a live drone feed
demands. Numbers to expect in a VM:

| | Host (M3 Max) | In the VM |
|---|---|---|
| Detector throughput | ~12 fps | ~5–9 fps |
| Publisher rate | — | 10 Hz (`rate:=`) |
| Frames dropped | — | yes, by design |

If it feels sluggish, lower the load rather than the expectations:

```bash
ros2 launch pennair_vision shapes.launch.py video:=... rate:=5 scale:=0.4
```

`scale` is safe to change: the intrinsics are scaled with the image, so depth does not move.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| VM boots to a black screen or won't start | x86 ISO, or **Emulate** instead of **Virtualize** | Recreate with the `arm64` ISO and Virtualize |
| **"Out of storage", or `df` shows `/` on `/cow`** | You are in the live "Try Ubuntu" session — Ubuntu was never installed, and `/` is a small RAM overlay. `/dev/sr0` and `/dev/loop0` at 100% are the read-only ISO and are always full; they are not the problem | Run **Install Ubuntu** from the live desktop, then `sudo poweroff` and remove the ISO |
| Reboots into the installer | ISO still attached | Shut down fully, then VM → Edit → CD/DVD drive → Clear → Save |
| `ImportError` mentioning NumPy, from `cv_bridge` | pip NumPy shadowing the system one | `pip uninstall numpy opencv-python`, then `sudo apt install --reinstall python3-numpy python3-opencv` |
| `ModuleNotFoundError: pose3d` | The nodes cannot find the repo root | `export PENNAIR_ROOT=$HOME/pennair`, or pass `repo_root:=$HOME/pennair` to the launch file |
| `package 'pennair_vision' not found` | Workspace not sourced in this terminal | `source ~/pennair/ros2_ws/install/setup.bash` |
| RViz crashes or renders nothing | Software OpenGL in a VM | `export LIBGL_ALWAYS_SOFTWARE=1` before launching, and set `rviz:=false` while debugging the pipeline |
| `/shapes/detections` is silent | No frames arriving | `ros2 topic hz /camera/image_raw`; if that is silent too, the video path is wrong — quote it, it has spaces |
| Markers all in one corner of RViz | `cx = cy = 0` in the supplied K | `principal_point:=center` |
| `plane_depth` is 0.0 | The circle has not been seen yet | Normal for the first few frames; if it persists, the circle is not being detected |

---

## Where to go next

- [README.md](README.md) — the algorithm, and [Part 5](README.md#part-5--ros-2) for the node
  graph and topic table.
- `ros2_ws/src/pennair_vision/pennair_vision/shape_detector.py` — the detector node; its
  per-frame body is six lines lifted from `detect_video_3d.py`.
