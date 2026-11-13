# Webots demo: traffic-sign detection with ground-truth distance

A camera car drives down a straight road past **all 6 sign classes** (Stop, Speed Limit 30/50, No Entry, Give Way, Keep Right) and
**3 look-alike decoys** (Speed Limit 20, Speed Limit 70, Turn Left), in 6 conditions: noon, overcast, dusk, night (headlights), fog, dense fog.
Because the simulator knows where every sign is, each frame gives the **true distance** and the **true box**, so we can score:

* detection rate of visible signs (5–40 m), per condition, and wrong-class predictions;
* false alarms (detections on decoys or on empty scenery);
* the pinhole distance estimate `Z = f · H / h` against the true distance → requirement **R6** in the notebook.

Lighter alternative to CARLA: one ~260 MB installer instead of 10–20 GB, runs on any NVIDIA/AMD GPU with OpenGL 3.3.

## Install (once)
1. Download **Webots R2025a** from <https://github.com/cyberbotics/webots/releases/tag/R2025a> (`webots-R2025a_setup.exe`, ~263 MB) and install it.
   Silent per-user install: `webots-R2025a_setup.exe /VERYSILENT /CURRENTUSER`.
2. The first run downloads the road and sign textures it needs (a few MB) and caches them.
3. Use the workshop Python environment (it has `ultralytics`, `torch`, `opencv-python`); nothing else to install.

## Run (from the `workshop` folder)
```bash
python webots_demo/run_webots.py                 # headless: 6 conditions, ~5-10 min on an RTX laptop GPU
python webots_demo/run_webots.py --gui --conditions noon night   # watch it: Webots window + a 'Detector view' window with the model's boxes
python webots_demo/run_webots.py --conditions noon night --video night_demo.mp4
```
`run_webots.py` regenerates `worlds/sign_drive.wbt`, writes the controller config and `runtime.ini` (so Webots runs the controller
with the same Python that launched the script), then starts Webots. Default weights: `models/signs6_y11s_fixed.pt` (`--weights` to change).

Outputs in `webots_demo/sim_results/`:

| File | Content |
|---|---|
| `weather_sweep.csv` | one row per condition: detection rate, signs detected, wrong class, false alarms, median / p90 distance error, detector p95 latency |
| `distance_log.csv` | every correct detection: true vs estimated distance |
| `observations.csv` | every visible sign in every frame: detected or not, predicted class |
| `demo_hd.mp4`, `demo.mp4` | annotated drive (noon, dusk, night, fog), 1280x720 (~160 MB) and a 960x540 copy (~30 MB): green = true sign box, orange = detection with estimated / true distance, red = false alarm |

`--gui` runs write to `sim_results_gui/` instead, so watching the demo never overwrites the measurement.
In the Webots window the main 3D view follows the car from behind; the small overlay is the car camera.

For the notebook: upload the `sim_results/` folder to Colab (`/content/sim_results`) or Drive (`traffic_sign_data/sim_results`); section 6.4 reads it and fills R6.

## Results with `signs6_y11s_fixed.pt` (RTX 5050, 2026-11-13)

| Condition | Visible-sign detection rate | Signs found | Wrong class | False alarms | Median / p90 distance error |
|---|---|---|---|---|---|
| noon | 0.63 | 6/6 | 1 | 1 | 3.9% / 9.9% |
| overcast | 0.64 | 6/6 | 4 | 2 | 4.9% / 10.6% |
| dusk | 0.64 | 6/6 | 1 | 0 | 9.0% / 16.1% |
| night (headlights) | 0.58 | 6/6 | 0 | 0 | 11.5% / 18.5% |
| fog | 0.64 | 6/6 | 7 | 1 | 6.6% / 11.4% |
| dense fog | 0.57 | 6/6 | 3 | 1 | 6.4% / 11.1% |

Overall median distance error **6.9%** (R6 limit 15%), from 1,466 correct detections (out of 2,376 sign-frames visible at 5–40 m); 5 false alarms in 5,436 frames.
Every sign is found in every condition, but **Speed Limit 30/50 are only recognised up close** (about 20–35% of the frames in which they are visible),
and the 16 wrong-class frames are all Speed Limit 30 read as 50 (the digits are a few pixels wide at 30–40 m). Give Way and No Entry are found almost every frame.
The detection rate counts every frame between 5 and 40 m, so 0.6 does not mean 40% of signs are missed: it means far signs are picked up late.
Good discussion points: higher input resolution or a second-stage classifier for speed digits, and the sim-to-real gap below.

The first run of a fresh install may report 0 detections if the sign textures are still downloading; the controller now waits 150 steps before driving.

## Files
* `run_webots.py` – world generator + launcher (sign layout, sizes and textures are in `SIGNS` / `GEOM`).
* `controllers/sign_drive/sign_drive.py` – the car: moves itself 0.5 m per frame (≈54 km/h at 30 fps), sets the lighting, runs YOLO, scores against ground truth.
* `worlds/sign_drive.wbt` – generated world (road, plain buildings, signs from the Webots traffic library with European textures).

## Limits (say them when presenting)
* **Sim-to-real gap:** rendered signs are clean and evenly lit; a real camera adds blur, glare, dirt and occlusion. Simulation is necessary evidence, not sufficient.
* The road is straight, there is no traffic, no rain rendering, and the car's motion is kinematic (no vehicle physics).
* The pinhole estimate uses the true panel height of each sign type (the "standard sign size" assumption). Real signs come in several sizes, which adds error.
* The Give Way texture is the Webots yield sign (red triangle with a white centre), slightly different from the European style in our training data.
