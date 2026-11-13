# %% [markdown]
# # From Prototype to Production: Computer Vision for Autonomous Cars
# A hands-on guide: you take one working model and walk it through the six steps a real team goes through before a model is allowed into a car.
#
# **The model you will work on:** a traffic-sign detector for 6 classes (Stop, Speed Limit 30, Speed Limit 50, No Entry, Give Way, Keep Right).
# * **v1, the prototype** (*SignSight ADAS*): YOLOv8n trained on one small dataset. Treat it as the model "currently in the car".
# * **The candidate:** YOLO11s trained on that dataset **plus** the European *Street Sign Set*, with corrected labels.
#
# The question that runs through the whole notebook: *is the candidate good enough to replace v1 on a real road, and how would you prove it?*
# The answer is **not** one accuracy number. By the end you will have a requirement list, a release gate and a scorecard that say SHIP or NO-SHIP, and why.
#
# | Step | Topic | What you do in the notebook |
# |---|---|---|
# | 1 | Requirements & metrics | Write testable requirements, build the metrics from scratch, price the errors |
# | 2 | Data | Audit labels, coverage, imbalance, leakage; version the dataset |
# | 3 | Validation | Train, pick an operating point, evaluate by slice, failure mining, **release gate** |
# | 4 | Deployment | Latency budget, ONNX / FP16 / INT8, re-validate after optimising |
# | 5 | Perception stack | Tracking, calibration & distance, plausibility checks, fallback states |
# | 6 | Monitoring & MLOps | Drift detection, shadow mode, labelling queue, model registry, scorecard |
# | 6.4 | Simulation | Webots drive with ground-truth distance (`webots_demo/`; CARLA scripts in `carla_demo/` also work) |
#
# **How to use this notebook**
# * Run the cells **top to bottom**; later steps reuse results from earlier ones. If a cell fails, fix it and re-run from there.
# * Each step starts with **what you will learn** and ends with **check yourself** questions. Try to answer them before reading on.
# * Read the outputs, not just the code: every table and plot is there to answer a question.
#
# **Credits.** Prototype adapted from the *SignSight ADAS* notebook by Malak Chartaoui (ENISo). Extra data: *Street Sign Set* by Alessandro Ferrante (CC BY 4.0).
# The requirements and costs in this notebook are **illustrative**, chosen for learning, not taken from a real safety case.

# %% [markdown]
# ## 0. Setup
# **Before you start**
# 1. In Colab: *Runtime → Change runtime type → T4 GPU*. Without a GPU, training is far too slow; use pre-trained weights instead.
# 2. **Data** (`DATA_SOURCE` in the config cell): `"kaggle"` downloads both datasets with your own Kaggle API token (*kaggle.com → Settings → API → Create New Token*).
#    Store it as a Colab secret named `KAGGLE_API_TOKEN` (key icon in the left bar), or paste it when the notebook asks; never write a token into a notebook cell.
#    `"drive"` reads them from `MyDrive/traffic_sign_data/a` and `/b` if someone shared them with you.
#    The corrected labels (`label_fixes.csv`) come with the repository and are applied automatically.
# 3. **Weights:** the trained models are in the repository's `models/` folder and are picked up automatically, so nothing is trained.
#    Training the candidate yourself takes well over an hour on a T4; to do it, set `WEIGHTS_MAIN = None` in the config cell. A good exercise for later.
#
# The first cell downloads the repository (in Colab), installs the libraries (Ultralytics, ONNX, ONNX Runtime) and prints which GPU you have.

# %%
import subprocess, sys, os, shutil, random, time, json, math, hashlib, csv, warnings
from pathlib import Path
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from itertools import combinations

warnings.filterwarnings("ignore")

# In Colab: download this workshop's repository (trained weights, label fixes, simulation results) and work inside it.
REPO_URL = "https://github.com/Moetez-Fradi/Eniso-Auto-Crew-CV-workshop"     # <- the workshop repository
REPO_DIR = Path("/content/Eniso-Auto-Crew-CV-workshop")
if Path("/content").exists() and not Path("models").exists():
    if not REPO_DIR.exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=False)
    if REPO_DIR.exists():
        os.chdir(REPO_DIR)
print("working folder:", Path.cwd())


def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


pip_install("ultralytics", "onnx", "onnxslim")
import torch
HAS_GPU = torch.cuda.is_available()
try:
    pip_install("onnxruntime-gpu" if HAS_GPU else "onnxruntime")
except Exception as e:                      # fall back to the CPU build
    print("onnxruntime-gpu install failed, using CPU build:", e)
    pip_install("onnxruntime")

import numpy as np, cv2, pandas as pd, yaml
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import linear_sum_assignment

os.environ["WANDB_DISABLED"] = "true"
print("GPU:", torch.cuda.get_device_name(0) if HAS_GPU else "none (CPU) -> training will be very slow, load weights instead")

# %%
# ----------------------------- CONFIG (edit here) -----------------------------
SEED = 42
TARGET_CLASSES = ["Stop", "Speed Limit 30", "Speed Limit 50", "No Entry", "Give Way", "Keep Right"]
NC = len(TARGET_CLASSES)
STOP = TARGET_CLASSES.index("Stop")

# Data: 'kaggle' (download with the Kaggle API) | 'drive' (folders on Google Drive) | 'local' (any folders)
DATA_SOURCE = os.environ.get("WS_DATA_SOURCE", "kaggle")
DATA_ROOT_DIR = Path(os.environ.get("WS_DATA_ROOT", "/content/data"))            # each dataset goes to DATA_ROOT_DIR/<tag>
DRIVE_DATA_DIR = Path("/content/drive/MyDrive/traffic_sign_data")                 # same layout on Drive: .../a and .../b
DATASETS = {   # tag -> source. Filenames get the tag as prefix, so every result can be sliced by source.
    "a": dict(name="SignSight (original prototype data)", kaggle="ahemateja19bec1025/trafficsignlocalizationdetectionyoloannotated",
              dir=os.environ.get("WS_DATA_A")),
    "b": dict(name="Street Sign Set (Ferrante 2025, CC BY 4.0)", kaggle="ferrantealessandro/street-sign-set",
              hf="AlessandroFerrante/StreetSignSet", dir=os.environ.get("WS_DATA_B")),
}
USE_DATASETS = ["a", "b"]    # ["a"] = the original prototype data only
# Oversampling in the training list: rare positives, and look-alike signs (source class names) that the model confuses with ours
OVERSAMPLE_POS = {"Keep Right": 3}
OVERSAMPLE_NEG = (["warn_left_curve", "warn_right_curve", "warn_sharp_left_curve", "warn_sharp_right_curve",
                   "forb_speed_over_20", "forb_speed_over_40", "mand_go_right", "mand_go_straight_right"], 2)
# Label corrections (a reviewed, versioned layer on top of the raw data, see 2.6). Searched in this order; skipped with a warning if absent.
LABEL_FIXES_CANDIDATES = [os.environ.get("WS_LABEL_FIXES", ""), "label_fixes.csv", "/content/label_fixes.csv",
                          "/content/drive/MyDrive/traffic_sign_data/label_fixes.csv"]
# Source class names -> our class. 'Turn Right' in dataset a is mostly the blue *keep right* sign (see the label audit in 2.6).
CLASS_ALIASES = {"Stop": ["stop", "prio_stop"], "Speed Limit 30": ["speed limit 30", "forb_speed_over_30"],
                 "Speed Limit 50": ["speed limit 50", "forb_speed_over_50"], "No Entry": ["no entry", "forb_no_entry"],
                 "Give Way": ["give way", "prio_give_way"], "Keep Right": ["turn right", "mand_pass_right"]}

# Model
BASE_MODEL = "yolo11s.pt"    # candidate architecture (yolov8n.pt = smaller/faster, for weak embedded hardware)
IMGSZ = 640
EPOCHS = int(os.environ.get("WS_EPOCHS", 50))
BATCH = 16
# Trained weights: if the repository's models/ folder is here, they are used and training is skipped (training takes over 1.5 h on a T4).
# Set the paths yourself to use other weights, or set both to None to train from scratch.
_repo_w = lambda name: str(Path("models", name).resolve()) if Path("models", name).exists() else None
WEIGHTS_MAIN = os.environ.get("WS_WEIGHTS_MAIN") or _repo_w("signs6_y11s_fixed.pt")
TRAIN_MAIN = True                                    # only used when WEIGHTS_MAIN is empty
# "v1" = the currently deployed model = the original prototype: YOLOv8n trained on dataset 'a' only (the provided weights were trained at 640 px).
WEIGHTS_V1 = os.environ.get("WS_WEIGHTS_V1") or _repo_w("signs6_v8n_original.pt")
TRAIN_V1_BASELINE = True
V1_BASE, V1_EPOCHS = "yolov8n.pt", int(os.environ.get("WS_V1_EPOCHS", 8))
V1_IMGSZ = int(os.environ.get("WS_V1_IMGSZ", 640 if WEIGHTS_V1 else 416))
print("candidate weights:", WEIGHTS_MAIN or "none -> will train", "| v1 weights:", WEIGHTS_V1 or "none -> will train")

# Optional / slow features
TRY_TENSORRT = False         # GPU only; adds a few minutes of setup

# Paths
IN_COLAB = Path("/content").exists()
WORK = Path("/content/work") if IN_COLAB else Path("./work").resolve()
WORK.mkdir(parents=True, exist_ok=True)
OUT = WORK / "signs6"        # prepared dataset (YOLO format, 6 classes)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = 0 if HAS_GPU else "cpu"
if not HAS_GPU:
    EPOCHS, V1_EPOCHS = min(EPOCHS, 3), min(V1_EPOCHS, 2)
    print("CPU only: epochs reduced. Set WEIGHTS_MAIN / WEIGHTS_V1 to use pre-trained weights.")

# %%
# ----------------------------- helpers -----------------------------
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
PALETTE = {i: tuple(int(v) for v in c) for i, c in enumerate(np.random.RandomState(SEED).randint(60, 255, (NC, 3)))}
_trapz = getattr(np, "trapezoid", None) or np.trapz


def show_grid(imgs, titles=None, cols=4, size=3.5):
    rows = math.ceil(len(imgs) / cols)
    plt.figure(figsize=(cols * size, rows * size))
    for i, im in enumerate(imgs):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis("off")
        if titles:
            plt.title(titles[i], fontsize=8)
    plt.tight_layout(); plt.show()


def draw_boxes(img, items, color=(0, 255, 0), thickness=2):
    """items: list of dict(box=(x1,y1,x2,y2), label=str)"""
    img = img.copy()
    for it in items:
        x1, y1, x2, y2 = map(int, it["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if it.get("label"):
            cv2.putText(img, it["label"], (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return img


def yolo_to_xyxy(x, y, w, h, W, H):
    """normalised YOLO centre/size -> pixel corners"""
    return (x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H


def quality(img):
    """-> grayscale image, brightness (mean grey level), sharpness (variance of the Laplacian)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray, float(gray.mean()), float(cv2.Laplacian(gray, cv2.CV_64F).var())


def nan0(v):
    return 0.0 if v != v else v


def sha_short(data: bytes, n=10):
    return hashlib.sha256(data).hexdigest()[:n]


def file_sha(path, n=10):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def finish_plot(name=None):
    plt.tight_layout()
    if name:
        plt.savefig(WORK / f"{name}.png", dpi=130)
    plt.show()
