# CARLA demo: YOLO traffic-sign detection with ground truth

Closing demo of the workshop (Step 5/6 bonus). A simulated car drives through CARLA, a camera feeds the detector, and because the simulator *knows* where every sign is, we can score the system with **true distances** and under **controlled weather / lighting**, which a static dataset cannot give.

| Script | What it does | Output |
|---|---|---|
| `carla_yolo_demo.py` | One drive, live window and/or mp4: detection box, class, confidence, **estimated vs true distance** | `carla_results/distance_log.csv`, `weather_sweep.csv` (1 row), optional `demo.mp4` |
| `carla_weather_sweep.py` | The same drive repeated in 7 conditions (noon, cloudy, wet, hard rain, sunset, night, fog), scored against ground truth | `carla_results/weather_sweep.csv`, `distance_log.csv` |
| `carla_common.py` | Shared code (camera, projection, matching, scoring) | – |

The notebook's section **6.4** reads `carla_results/` and fills requirement **R6** (distance error).

> **Status:** the geometry, matching and scoring helpers are unit-tested without the simulator, but the scripts have **not been run against a real CARLA server yet**. Do one full dry run on the demo machine **before** the workshop (budget 1 hour for install + first run), and record a fallback video (see below).

## 1. What you need

* A PC with an **NVIDIA GPU** (6 GB works at low quality, 8 GB+ recommended) and ~30 GB disk. CARLA does **not** run on Google Colab (it needs a rendering GPU and a running server). Run it on a club PC / laptop with GPU.
* Windows 10/11 or Ubuntu 20.04/22.04. Native install is the easiest on Windows; Docker is practical on Linux.
* Python that matches the `carla` wheel (CARLA 0.9.15 ships wheels for Python 3.8–3.10; check the CARLA docs for the version you download).

## 2. Install

1. Download a CARLA release from the official site / GitHub releases (**0.9.15** is a safe choice) and unzip it.
2. Create an environment and install the client:
   ```bash
   python -m venv carla-env && source carla-env/bin/activate      # Windows: carla-env\Scripts\activate
   pip install -r requirements.txt
   ```
   If `pip install carla==0.9.15` finds no wheel for your Python, install the one shipped inside CARLA: `PythonAPI/carla/dist/carla-*.whl`.
3. Use the trained model `../models/signs6_y11s_fixed.pt` (pass it with `--weights`), or use the default COCO `yolov8n.pt` (it only knows **Stop signs**, but works out of the box, see section 5).

## 3. Run

Start the server first (low quality = faster, less GPU memory):

```bash
# Windows
CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600
# Linux
./CarlaUE4.sh -quality-level=Low
# Linux, no window (server only)
./CarlaUE4.sh -quality-level=Low -RenderOffScreen
# Docker (needs nvidia-container-toolkit; see the CARLA docs for your version)
docker run --privileged --gpus all --net=host carlasim/carla:0.9.15 /bin/bash ./CarlaUE4.sh -RenderOffScreen
```

Then, from this folder:

```bash
# 1) is the setup alive? list the signs of the map (should print > 0)
python carla_yolo_demo.py --list-signs

# 2) the live demo (ESC to stop) + a recording
python carla_yolo_demo.py --weather ClearNoon --frames 600 --show --save demo.mp4

# 3) your trained model, at night
python carla_yolo_demo.py --weights best.pt --weather Night --frames 600 --show

# 4) the full sweep (about 2-4 minutes per weather on a decent GPU)
python carla_weather_sweep.py --weights best.pt --frames 400
```

The first launch of a map takes a while (the server compiles shaders). The scripts use **synchronous mode** with a fixed 0.05 s step, so results do not depend on your GPU speed.

## 4. First-run checklist (do this once)

* **Are the ground-truth markers right?** With `--show`, green dots (ground truth) must sit on or just above signs that face the car. If dots appear on signs seen from behind and not on the visible ones, flip the readable side: `--sign-front backward` (or `any` to disable the check).
* **Are signs found?** If `--list-signs` prints 0, the map has no sign actors of the supported types: try `--town Town10HD_Opt` or `Town01`.
* **Does the car move along signs?** `--spawn -1` (default) picks the spawn point with the most signs ahead; pass `--spawn N` to fix a drive.
* **Distance bias?** The estimate uses an assumed real sign height (`--sign-h 0.75`). If errors are one-sided (always too near / too far), the assumption is the cause: that is the lesson of Step 5.1. The camera focal length is *exact* here (taken from the simulator's FOV), so the error you see is height assumption + box error.

## 5. What to expect and how to explain it

* **Domain gap.** The notebook model was trained on photos of real signs; CARLA signs are rendered, with a different look. Expect lower detection rates than on the test set. That is the point: *a good score on the dataset does not transfer automatically.* Options to discuss: fine-tune on simulator images, collect real footage, domain randomisation.
* **COCO `yolov8n.pt` as a fallback.** It detects "stop sign" in CARLA reasonably well and needs no training: good for a guaranteed live demo. Only the `stop` family is scored in that case.
* **Detection rate is a lower bound.** Ground-truth signs that are partly occluded (trees, vehicles) still count as "visible"; unmatched detections include signs that failed the facing/range filters.
* **Weather sweep = Step 3 slices in the simulator.** The same car, route and seed in 7 conditions: any change in detection rate is caused by the condition, not by the data.

## 6. Fallback plan for the workshop

1. Record `demo.mp4` and a `carla_results/` folder **the day before** (steps above).
2. If the live demo fails: play the video, then upload `carla_results/` to the notebook and run section 6.4.
3. If CARLA cannot be installed on the demo machine at all, show the pipeline demo from notebook section 5.6 and walk through this README as "what we would do next".

## 7. Other tools worth knowing (for the "technologies available" slide)

| Tool | Use it for | Notes |
|---|---|---|
| **CARLA** | Photoreal driving simulation, sensors (camera/LiDAR/radar), traffic, weather | Heaviest GPU requirement; Python API; ROS 2 bridge available |
| **Webots** | Easy robot/vehicle simulation | Lighter; good for small-scale car platforms |
| **Gazebo + ROS 2** | Robotics integration, custom small vehicle models | Standard in robotics; less photoreal |
| **Duckietown / Donkey Car** | Tiny real/sim autonomous cars | Cheap, educational |
| **NVIDIA Isaac Sim / DRIVE Sim** | Synthetic data at scale, sensor simulation | Needs a strong RTX GPU |
| **ROS 2** | Message-passing between camera, perception, planning, control nodes | The integration layer on most research vehicles |
| **ONNX Runtime / TensorRT / OpenVINO** | Fast inference on target hardware | Step 4 of the notebook |
| **CVAT / Roboflow / Label Studio** | Labelling and dataset management | Step 2 and the data flywheel |
| **MLflow / Weights & Biases / DVC** | Experiment tracking, data and model versioning | Step 6 (registry) |
