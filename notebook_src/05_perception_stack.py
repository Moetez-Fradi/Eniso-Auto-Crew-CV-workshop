# %% [markdown]
# # Step 5 — The perception stack
# **What you will learn:** what has to be built around a detector before its output can drive a car.
#
# A detector is one box in a longer chain. Production systems add **temporal filtering** (tracking), **geometry** (calibrated distance), **cross-checks** (maps, other sensors) and **fallbacks** (what to do when perception is unreliable).
#
# `camera -> detector -> tracker -> distance -> plausibility check -> health monitor -> decision`
#
# The four ideas in this step:
# * one detection is not one sign: detections flicker from frame to frame, so they are tracked over time;
# * distance needs a **calibrated** camera, not a guessed field of view;
# * a map prior catches readings that make no sense;
# * the system must *know* when it cannot see, and fall back to something safer.

# %% [markdown]
# ## 5.1 Distance from a single camera — and how wrong it can be
# Pinhole model: **Z = f · H / h** (focal length in pixels × real sign height ÷ height in pixels).
# Three error sources: a wrong assumed **sign height**, a wrong **focal length** (field of view guessed instead of calibrated), and **±2 px** of box error — which explodes with distance because far signs are only a few pixels tall.
#
# **What to check:** in the plot, at what distance does the ±2 px error alone break the 15% limit of R6?

# %%
@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_hfov(cls, width, height, hfov_deg):
        f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
        return cls(width, height, f, f, width / 2, height / 2)

    def distance(self, h_px, real_h):
        return self.fy * real_h / max(h_px, 1e-6)

    def height_px(self, dist, real_h):
        return self.fy * real_h / dist


TRUE_CAM = Camera(640, 480, 800.0, 800.0, 320.0, 240.0)           # the real camera (hfov ~ 43.6 deg)
ASSUMED = Camera.from_hfov(640, 480, 60)                          # a typical "guess"
REAL_H, ASSUMED_H = 0.60, 0.75
Z = np.linspace(5, 50, 100)
h_true = TRUE_CAM.height_px(Z, REAL_H)
plt.figure(figsize=(7, 3.4))
plt.plot(Z, np.full_like(Z, (ASSUMED_H / REAL_H - 1) * 100), label=f"sign height assumed {ASSUMED_H} m instead of {REAL_H} m")
plt.plot(Z, np.full_like(Z, (ASSUMED.fy / TRUE_CAM.fy - 1) * 100), label="focal length from guessed 60 deg FOV (true 43.6 deg)")
plt.plot(Z, (TRUE_CAM.fy * REAL_H / np.maximum(h_true - 2, 0.5) / Z - 1) * 100, label="box 2 px too small")
for sign in (1, -1):
    plt.axhline(sign * REQS[5].target, color="k", ls=":")
plt.xlabel("true distance (m)"); plt.ylabel("distance error (%)"); plt.legend(fontsize=7); plt.title("Where distance errors come from (dotted = R6 limit)")
finish_plot("distance_sensitivity")

# %% [markdown]
# ## 5.2 Calibration fixes the focal-length error
# The cell simulates photographing a chessboard from 14 poses with a camera of known focal length (800 px), adds 0.3 px of corner noise, and lets OpenCV recover it.
# On a real car you do exactly this once per camera, with a printed board. Compare the recovered focal length with the true one.

# %%
def simulate_calibration(K_true, n_views=14, noise=0.3, seed=SEED, size=(640, 480)):
    rng = np.random.RandomState(seed)
    cols, rows, sq = 9, 6, 0.025
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq
    objpts, imgpts = [], []
    while len(objpts) < n_views:
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-0.15, -0.05), rng.uniform(-0.1, -0.02), rng.uniform(0.5, 0.9)])
        p = cv2.projectPoints(objp, rvec, tvec, K_true, np.zeros(5))[0].reshape(-1, 2)
        if ((p < 5) | (p > np.array(size) - 5)).any():          # board must be fully inside the image
            continue
        objpts.append(objp); imgpts.append((p + rng.normal(0, noise, p.shape)).astype(np.float32).reshape(-1, 1, 2))
    rms, K, *_ = cv2.calibrateCamera(objpts, imgpts, size, None, None)
    return rms, K


K_true = np.array([[TRUE_CAM.fx, 0, TRUE_CAM.cx], [0, TRUE_CAM.fy, TRUE_CAM.cy], [0, 0, 1.0]])
rms, K_est = simulate_calibration(K_true)
CALIB_CAM = Camera(640, 480, float(K_est[0, 0]), float(K_est[1, 1]), float(K_est[0, 2]), float(K_est[1, 2]))
print(f"reprojection RMS = {rms:.2f} px | focal estimate fx = {CALIB_CAM.fx:.1f} px (true 800.0, error {abs(CALIB_CAM.fx / 800 - 1) * 100:.1f}%)")
z_true, h_px = 20.0, TRUE_CAM.height_px(20.0, REAL_H)
for nm, cam in [("guessed 60 deg FOV", ASSUMED), ("calibrated", CALIB_CAM)]:
    d = cam.distance(h_px, REAL_H)
    print(f"  {nm:20s}: sign really at {z_true:.0f} m -> estimated {d:.1f} m  ({(d / z_true - 1) * 100:+.1f}%)")

# %% [markdown]
# ## 5.3 Tracking — one sign, one identity
# A raw detector flickers: a sign is found in frame 1, lost in frame 2, found again in frame 3. A **Kalman filter** predicts where the box should be, **Hungarian matching** links detections to tracks,
# a track is *confirmed* only after 3 hits, and the class is decided by a **confidence-weighted vote** over the last 10 frames (so one wrong frame cannot flip "Speed 30" into "Speed 50").

# %%
def box_to_z(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]], float)


def z_to_box(z):
    cx, cy, w, h = z[:4]
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class KalmanBox:
    """Constant-velocity Kalman filter on [cx, cy, w, h, vx, vy, vw, vh]."""
    def __init__(self, box, meas_std=2.0):
        self.x = np.concatenate([box_to_z(box), np.zeros(4)])
        self.P = np.diag([10.0] * 4 + [100.0] * 4)
        self.F = np.eye(8); self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)
        self.Q = np.diag([1.0] * 4 + [0.5] * 4) * 0.1
        self.R = np.eye(4) * meas_std ** 2

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return z_to_box(self.x)

    def update(self, box):
        y = box_to_z(box) - self.H @ self.x
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P


class Track:
    _next = 0

    def __init__(self, det):
        Track._next += 1
        self.id = Track._next
        self.kf, self.box = KalmanBox(det["box"]), det["box"]
        self.votes = deque([(det["cls"], det["conf"])], maxlen=10)
        self.hits, self.misses, self.dist = 1, 0, None

    def predict(self):
        self.box = self.kf.predict()
        return self.box

    def update(self, det):
        self.kf.update(det["box"]); self.box = z_to_box(self.kf.x)
        self.votes.append((det["cls"], det["conf"])); self.hits += 1; self.misses = 0

    @property
    def label(self):
        w = Counter()
        for c, s in self.votes:
            w[c] += s
        cls = w.most_common(1)[0][0]
        return cls, float(np.mean([s for c, s in self.votes if c == cls]))


class SignTracker:
    def __init__(self, iou_gate=0.1, max_age=6, min_hits=3):
        self.tracks, self.iou_gate, self.max_age, self.min_hits = [], iou_gate, max_age, min_hits

    def step(self, dets):
        preds = [t.predict() for t in self.tracks]
        matched_t, matched_d = set(), set()
        if self.tracks and dets:
            cost = np.array([[1 - iou_xyxy(p, d["box"]) for d in dets] for p in preds])
            for r, c in zip(*linear_sum_assignment(cost)):
                if 1 - cost[r, c] >= self.iou_gate:
                    self.tracks[r].update(dets[c]); matched_t.add(r); matched_d.add(c)
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.misses += 1
        self.tracks += [Track(d) for j, d in enumerate(dets) if j not in matched_d]
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        return [t for t in self.tracks if t.hits >= self.min_hits and t.misses <= 1]


# tiny check: a detector that drops 40% of frames and sometimes mislabels -> tracker output stays stable
rng = np.random.RandomState(1)
tk, trace = SignTracker(), []
for f in range(40):
    box = (200 + 3 * f, 150 + f, 240 + 4 * f, 190 + 4 * f)
    dets = [] if rng.rand() < 0.4 else [{"box": tuple(b + rng.normal(0, 1.5) for b in box), "cls": 2 if rng.rand() < 0.15 else 1, "conf": 0.6 + 0.3 * rng.rand()}]
    out = tk.step(dets)
    trace.append((len(dets), len(out), TARGET_CLASSES[out[0].label[0]] if out else "-"))
print("frames with a raw detection:", sum(t[0] for t in trace), "/ 40  | frames with a stable track:", sum(t[1] for t in trace), "/ 40")
print("class shown by the track:", Counter(t[2] for t in trace))

# %% [markdown]
# ## 5.4 Plausibility check against a map prior
# The camera says "Speed Limit 30", but the map says this road is 50 and a 120 sign in a 30 zone makes no sense. Production systems **fuse** vision with other sources and flag disagreement instead of blindly obeying the camera.

# %%
def speed_of(cls_name):
    return int(cls_name.split()[-1]) if cls_name.startswith("Speed Limit") else None


def plausibility(detected_kmh, map_kmh, max_step=30):
    if map_kmh is None:
        return "NO_MAP"
    if detected_kmh == map_kmh:
        return "CONSISTENT"
    return "ZONE_CHANGE_PLAUSIBLE" if abs(detected_kmh - map_kmh) <= max_step else "SUSPECT"


pd.DataFrame([(d, m, plausibility(d, m)) for d, m in [(50, 50), (30, 50), (120, 30), (50, None)]],
             columns=["sign reads (km/h)", "map says (km/h)", "verdict"])

# %% [markdown]
# ## 5.5 Knowing when you cannot see: a health monitor with fallback states
# If the image is too dark or too blurry, detections cannot be trusted — *and the absence of detections is not "no signs"*. A small state machine tracks image health with hysteresis (no flapping):
#
# | State | Meaning | Action |
# |---|---|---|
# | NOMINAL | image quality inside the ODD | use camera signs |
# | DEGRADED | dark / blurry | lower trust, prefer map speed limit |
# | FAILSAFE | very dark for a long time | ignore camera, alert the driver |

# %%
class PerceptionHealth:
    ACTIONS = {"NOMINAL": "use camera signs", "DEGRADED": "prefer map limit, lower trust", "FAILSAFE": "ignore camera, alert driver"}

    def __init__(self, dark_thr, blur_thr, fail_thr, to_degraded=3, to_failsafe=8, to_nominal=10):
        self.dark_thr, self.blur_thr, self.fail_thr = dark_thr, blur_thr, fail_thr
        self.cfg = (to_degraded, to_failsafe, to_nominal)
        self.state, self.bad, self.good, self.very_bad = "NOMINAL", 0, 0, 0

    def update(self, brightness, blur):
        bad = brightness < self.dark_thr or blur < self.blur_thr
        self.bad, self.good = (self.bad + 1, 0) if bad else (0, self.good + 1)
        self.very_bad = self.very_bad + 1 if brightness < self.fail_thr else 0
        d, f, n = self.cfg
        if self.state == "NOMINAL" and self.bad >= d:
            self.state = "DEGRADED"
        elif self.state == "DEGRADED" and self.very_bad >= f:
            self.state = "FAILSAFE"
        elif self.state in ("DEGRADED", "FAILSAFE") and self.good >= n:
            self.state = "NOMINAL"
        return self.state

# %% [markdown]
# ## 5.6 Put it together on a video
# The demo video **simulates a drive**: it zooms into 8 test images (the car approaches the sign) and fades the light progressively in the second half (dusk → night).
# Watch the *confirmed* tracks, the distance, and the health monitor moving from NOMINAL to DEGRADED and FAILSAFE. The detection log (`detections_log.csv`) records every frame.
# *(This is a synthetic stand-in: real footage gives more convincing results, and the simulation in section 6.4 gives distance **ground truth**.)*

# %%
def build_demo_video(path, n_clips=8, frames=24, size=640, fps=20, fade_start=0.5, min_brightness=0.12):
    pool = [it for it in ITEMS["test"] if it["gts"]]
    clips = random.Random(SEED).sample(pool, min(n_clips, len(pool)))
    total = len(clips) * frames
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    n = 0
    for it in clips:
        img = cv2.imread(it["path"]); H, W = img.shape[:2]
        b = it["gts"][0]["box"]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        for k in range(frames):                                   # slow zoom towards the sign
            s = 1 + 0.8 * k / (frames - 1)
            cw, ch = W / s, H / s
            x0, y0 = int(np.clip(cx - cw / 2, 0, W - cw)), int(np.clip(cy - ch / 2, 0, H - ch))
            fr = cv2.resize(img[y0:y0 + int(ch), x0:x0 + int(cw)], (size, size))
            p = n / total
            if p > fade_start:                                    # second half: fade to dark
                fr = degrade(fr, "brightness", 1 - (1 - min_brightness) * (p - fade_start) / (1 - fade_start))
            wr.write(fr); n += 1
    wr.release()
    return Path(path)


def run_pipeline(video, detector, thr, camera, out_path, csv_path, sign_h=0.60, map_limit=50, warn_dist=10.0):
    cap = cv2.VideoCapture(str(video))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 20
    wr = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (W, H))
    Track._next = 0
    tracker = SignTracker()
    health = PerceptionHealth(dark_thr=BR_LO * 0.7, blur_thr=BL_LO * 0.1, fail_thr=BR_LO * 0.35)   # thresholds are tunable; zoomed demo frames are soft
    log = [("frame", "time_s", "track_id", "class", "confidence", "x1", "y1", "x2", "y2", "distance_m", "map_check", "health")]
    timeline, preview, f = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _, br, bl = quality(frame)
        state = health.update(br, bl)
        t0 = time.perf_counter()
        dets = detector(frame, thr)
        tracks = tracker.step(dets)
        lat = (time.perf_counter() - t0) * 1000
        warn = False
        for t in tracks:
            cls, conf = t.label
            name = TARGET_CLASSES[cls]
            d_new = camera.distance(t.box[3] - t.box[1], sign_h)
            t.dist = d_new if t.dist is None else 0.7 * t.dist + 0.3 * d_new       # smooth the estimate
            check = plausibility(speed_of(name), map_limit) if speed_of(name) else ""
            if state != "FAILSAFE":
                x1, y1, x2, y2 = map(int, t.box)
                col = PALETTE[cls]
                txt = f"#{t.id} {name} {conf * 100:.0f}% {t.dist:.1f}m" + (f" [{check}]" if check else "")
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                for c, th in [((0, 0, 0), 3), (col, 1)]:                          # dark outline, then coloured text
                    cv2.putText(frame, txt, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, th, cv2.LINE_AA)
                warn = warn or (cls == STOP and t.dist < warn_dist)
            log.append((f, round(f / fps_in, 3), t.id, name, round(conf, 3), *map(int, t.box), round(t.dist, 2), check, state))
        scol = {"NOMINAL": (0, 200, 0), "DEGRADED": (0, 165, 255), "FAILSAFE": (0, 0, 255)}[state]
        cv2.putText(frame, f"{state}: {health.ACTIONS[state]}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, scol, 2, cv2.LINE_AA)
        if warn:
            cv2.putText(frame, "STOP AHEAD", (W // 2 - 110, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
        wr.write(frame)
        timeline.append({"frame": f, "brightness": br, "blur": bl, "state": state, "raw_dets": len(dets), "tracks": len(tracks), "latency_ms": lat})
        if f % 30 == 0:
            preview.append(frame.copy())
        f += 1
    cap.release(); wr.release()
    with open(csv_path, "w", newline="") as fh:
        csv.writer(fh).writerows(log)
    return pd.DataFrame(timeline), preview


DEMO_IN = build_demo_video(WORK / "demo_input.mp4")
PIPE_DET = VARIANTS.get(DEPLOY_NAME, DET_MAIN)
CAM = Camera.from_hfov(640, 640, 60)                          # synthetic video: field of view is an assumption here
TL, preview = run_pipeline(DEMO_IN, PIPE_DET, OP_THR, CAM, WORK / "pipeline_raw.mp4", WORK / "detections_log.csv")
print(f"{len(TL)} frames | median latency {TL.latency_ms.median():.1f} ms | states:", dict(TL.state.value_counts()))
show_grid(preview[:8], cols=4, size=3.6)

# %%
fig, ax = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
ax[0].plot(TL.frame, TL.brightness, color="k"); ax[0].set_ylabel("brightness"); ax[0].axhline(BR_LO, color="gray", ls=":")
ax[1].plot(TL.frame, TL.raw_dets, label="raw detections", alpha=.6); ax[1].plot(TL.frame, TL.tracks, label="confirmed tracks", lw=2); ax[1].legend(fontsize=8); ax[1].set_ylabel("count")
cmap = {"NOMINAL": "tab:green", "DEGRADED": "tab:orange", "FAILSAFE": "tab:red"}
for st, sub in TL.groupby("state"):
    ax[2].scatter(sub.frame, [st] * len(sub), c=cmap[st], s=18)
ax[2].set_xlabel("frame"); ax[2].set_ylabel("health state")
plt.suptitle("Pipeline timeline: image quality, detection stability, health state", y=0.99); finish_plot("pipeline_timeline")

# %%
try:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(WORK / "pipeline_raw.mp4"), "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(WORK / "pipeline.mp4")], check=True)
    from IPython.display import Video
    display(Video(str(WORK / "pipeline.mp4"), embed=True, width=480))
except Exception as e:
    print("ffmpeg conversion unavailable:", e, "-> download", WORK / "pipeline_raw.mp4")
pd.read_csv(WORK / "detections_log.csv").head(12)

# %% [markdown]
# **Check yourself (Step 5)**
# 1. A sign is 20 px tall in the image and your box is off by 2 px. Roughly how large is the distance error? And for a sign 80 px tall?
# 2. Why does a track need 3 hits before it is confirmed? What does that cost at 15 frames per second?
# 3. The camera sees nothing for 10 seconds on a dark road. Does that mean there are no signs? What should the system do?
