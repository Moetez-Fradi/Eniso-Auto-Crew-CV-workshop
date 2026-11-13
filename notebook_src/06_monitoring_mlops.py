# %% [markdown]
# # Step 6 — Monitoring and MLOps
# **What you will learn:** how a model is watched and improved after release, and how every release stays traceable.
#
# Shipping is the *start* of the model's life. The world drifts (new seasons, cities, sign designs, camera ageing), so production systems
# 1. **monitor** the inputs and outputs without labels (drift detection),
# 2. run in **shadow mode** and collect disagreements and uncertain cases,
# 3. feed those back into labelling → retraining (the **data flywheel**, back to Step 2),
# 4. keep a **registry**: every model is traceable to its data, metrics and gate decision, promotion is controlled, rollback is one call.

# %% [markdown]
# ## 6.1 Drift monitoring without labels
# We replay the "field" as 6 weekly batches. From week 5 on the images are **darker and noisier** (a winter-evening shift the model never saw).
# Labels are not available in the real field, so the monitor only looks at **inputs** (brightness, sharpness) and **outputs** (confidence of detections), compared to the training reference with the
# **Population Stability Index** (PSI: < 0.1 stable, 0.1–0.25 moderate, > 0.25 major shift; the alert limit is raised automatically when a weekly window is too small to trust).
# In this simulation the truth *is* known, so you can check that the label-free alarm really tracks the real performance drop.
#
# **What to check:** in which week does the alarm fire, and in which week does the true F1 drop? Does any signal fire too late or not at all?

# %%
def psi(ref, cur, bins=5):
    edges = np.unique(np.percentile(ref, np.linspace(0, 100, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r, c = (np.clip(np.histogram(x, edges)[0] / len(x), 1e-3, None) for x in (ref, cur))
    return float(np.sum((c - r) * np.log(c / r)))


def psi_limit(ref, n, floor=0.25, trials=200):
    """Alert limit = max(0.25, 99th percentile of the PSI you get from pure sampling noise with n samples).
    Small windows are noisy: without this correction a monitor cries wolf every week."""
    r = np.random.RandomState(SEED)
    return max(floor, float(np.percentile([psi(ref, r.choice(ref, n, replace=True)) for _ in range(trials)], 99)))


def dusk(img):
    return degrade(degrade(img, "brightness", 0.45), "noise", 8)


FIELD_DET = PIPE_DET
pool = ITEMS["val"] + ITEMS["test"]
random.Random(SEED).shuffle(pool)
N_WEEKS, SHIFT_FROM = 6, 4
weeks = np.array_split(np.arange(len(pool)), N_WEEKS)

ref_bright = IMGS[IMGS.split == "train"].brightness.values
ref_blur = IMGS[IMGS.split == "train"].blur.values
ref_conf = np.array([p["conf"] for r in PACK["candidate"]["val"] for p in r["preds"] if p["conf"] >= OP_THR])

FIELD_RECS, rows = [], []
for w, idx in enumerate(weeks, start=1):
    recs = []
    for i in idx:
        it = pool[i]
        img = cv2.imread(it["path"])
        if w >= SHIFT_FROM:
            img = dusk(img)
        _, b, l = quality(img)
        recs.append({"path": it["path"], "gts": it["gts"], "preds": FIELD_DET(img, 0.01), "brightness": b, "blur": l, "week": w})
    FIELD_RECS += recs
    br, bl = [r["brightness"] for r in recs], [r["blur"] for r in recs]
    confs = np.array([p["conf"] for r in recs for p in r["preds"] if p["conf"] >= OP_THR])
    m = metrics_from(per_image_counts(recs, OP_THR))
    pb, pl = psi(ref_bright, br), psi(ref_blur, bl)
    pc = psi(ref_conf, confs) if (len(confs) >= 10 and len(ref_conf) >= 10) else float("nan")
    alert = pb > psi_limit(ref_bright, len(br)) or pl > psi_limit(ref_blur, len(bl)) or (pc == pc and pc > psi_limit(ref_conf, len(confs)))
    rows.append({"week": w, "psi_brightness": pb, "psi_blur": pl, "psi_conf": pc, "ALERT": bool(alert),
                 "ks_p_brightness": float(stats.ks_2samp(ref_bright, br).pvalue), "dets_per_img": len(confs) / len(recs),
                 "mean_conf": float(confs.mean()) if len(confs) else float("nan"),
                 "true_recall": m["recall"], "true_f1": nan0(m["f1"])})
DRIFT = pd.DataFrame(rows).set_index("week")
display(DRIFT.round(3))

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
for c in ["psi_brightness", "psi_blur", "psi_conf"]:
    ax[0].plot(DRIFT.index, DRIFT[c], marker="o", label=c)
ax[0].axhline(0.25, color="r", ls="--"); ax[0].set_title("Label-free monitors (PSI vs training)")
ax[1].plot(DRIFT.index, DRIFT.true_f1, marker="o", color="k", label="true F1 (needs labels)"); ax[1].plot(DRIFT.index, DRIFT.true_recall, marker="s", color="tab:red", label="true recall")
ax[1].set_ylim(0, 1); ax[1].set_title("Real performance (simulation only)")
for a in ax:
    a.axvspan(SHIFT_FROM - 0.5, N_WEEKS + 0.5, color="orange", alpha=.15); a.set_xlabel("week"); a.legend(fontsize=8)
finish_plot("drift")
first = DRIFT.index[DRIFT.ALERT][0] if DRIFT.ALERT.any() else None
print("First label-free alert: week", first, "| actual shift starts at week", SHIFT_FROM)

# %% [markdown]
# ## 6.2 Shadow mode and the labelling queue
# The model runs silently next to the "ground truth" source (here: our labels; in a car: the driver's behaviour, other sensors, human audits).
# Every **disagreement** and every **uncertain** prediction is queued for human labelling, ranked by priority — this is the data flywheel feeding Step 2.
# Look at the top of the queue: are these the images you would want labelled first?

# %%
def build_queue(records, thr, uncertain_band=0.20):
    q = []
    for r in records:
        m = match_image(r["preds"], r["gts"], thr)
        reasons, prio = [], 0
        wrong_gt = set(m["wrong"].values())
        for j, hit in enumerate(m["gt_hit"]):
            if hit:
                continue
            if j in wrong_gt:
                reasons.append("wrong class"); prio += 2
            else:
                reasons.append("model missed " + TARGET_CLASSES[r["gts"][j]["cls"]]); prio += 3 if r["gts"][j]["cls"] == STOP else 2
        for i in m["keep"]:
            if not m["pred_tp"][i] and i not in m["wrong"]:
                reasons.append("model saw a sign that is not labelled"); prio += 2
            elif r["preds"][i]["conf"] < thr + uncertain_band:
                reasons.append("low confidence"); prio += 1
        if reasons:
            prio += int(band_brightness(r["brightness"]) == "dark")
            q.append({"path": r["path"], "week": r.get("week"), "priority": prio, "reasons": "; ".join(sorted(set(reasons)))})
    return pd.DataFrame(q).sort_values("priority", ascending=False).reset_index(drop=True)


QUEUE = build_queue(FIELD_RECS, OP_THR)
QUEUE.to_csv(WORK / "labeling_queue.csv", index=False)
print(f"{len(QUEUE)} of {len(FIELD_RECS)} field images queued for labelling (saved to labeling_queue.csv)")
display(QUEUE.groupby("week").size().rename("queued images per week").to_frame().T)
top = QUEUE.head(8)
display(top)
show_grid([cv2.imread(p) for p in top.path], [f"prio {r.priority}: {r.reasons[:34]}" for r in top.itertuples()], cols=4, size=3.2)

# %% [markdown]
# ## 6.3 Model registry and controlled promotion
# A model is not "best.pt in somebody's Drive". It has an **ID**, the **hash of its weights**, the **dataset ID** it was trained on, its **metrics** and the **gate decision**.
# Promotion to production is allowed only if the gate said SHIP — anything else needs a **written waiver**, which is logged. Rollback is one call.
# (In real teams this lives in MLflow / W&B / an internal registry; the principle is the same.)
# The cell tries several promotions on purpose, including a blocked one; read each line of the log and check why it was allowed or refused.

# %%
class ModelRegistry:
    def __init__(self, path):
        self.path = Path(path)
        self.d = json.loads(self.path.read_text()) if self.path.exists() else {"models": [], "production": None, "history": [], "log": []}

    def _save(self):
        self.path.write_text(json.dumps(self.d, indent=2, default=str))

    def _log(self, msg):
        self.d["log"].append(f"{time.strftime('%H:%M:%S')}  {msg}")
        self._save()

    def register(self, name, weights, dataset_id, metrics, gate):
        mid = f"m{len(self.d['models']) + 1}-{file_sha(weights, 8)}"
        self.d["models"].append({"id": mid, "name": name, "weights": str(weights), "dataset_id": dataset_id,
                                 "metrics": {k: (None if v is None else round(float(v), 4)) for k, v in metrics.items()},
                                 "gate": gate, "created": time.strftime("%Y-%m-%d %H:%M:%S")})
        self._log(f"registered {mid} ({name}), gate={gate}")
        return mid

    def promote(self, mid, override_reason=None):
        e = next(m for m in self.d["models"] if m["id"] == mid)
        waived = e["gate"] != "SHIP"
        if waived and not override_reason:
            raise PermissionError(f"{mid}: gate decision is '{e['gate']}', promotion blocked (a written waiver is required)")
        if waived:
            e["waiver"] = override_reason
        if self.d["production"]:
            self.d["history"].append(self.d["production"])
        self.d["production"] = mid
        self._log(f"PROMOTED {mid}" + (f" with waiver: {override_reason}" if waived else ""))

    def rollback(self):
        if not self.d["history"]:
            raise RuntimeError("nothing to roll back to")
        bad, self.d["production"] = self.d["production"], self.d["history"].pop()
        self._log(f"ROLLBACK {bad} -> {self.d['production']}")


(WORK / "registry.json").unlink(missing_ok=True)
REG = ModelRegistry(WORK / "registry.json")
if WEIGHTS_V1:
    v1_id = REG.register("v1 (deployed)", WEIGHTS_V1, DATASET_ID, {k: PACK["v1"]["req"].get(k) for k in ["stop_recall", "other_min_recall", "bg_fa_per_100"]}, "LEGACY")
    REG.promote(v1_id, override_reason="initial deployment of the legacy baseline")
cand_id = REG.register("candidate", WEIGHTS_MAIN, DATASET_ID, PACK["candidate"]["req"], GATE_DECISION)
try:
    REG.promote(cand_id)
    print("candidate promoted to production")
except PermissionError as e:
    print("BLOCKED ->", e)
    print("(To ship anyway an engineer must write a waiver: REG.promote(cand_id, override_reason='...') - and it is logged forever.)")
print("\n".join(REG.d["log"]))
print("production model:", REG.d["production"])

# %% [markdown]
# ## 6.4 Simulation evidence: distance against ground truth (requirement R6)
# Simulation gives what the dataset cannot: **ground-truth distance** and **controlled weather / lighting**. It is the only evidence for R6 in this notebook.
# The results come from `webots_demo/` (Webots, open source, ~260 MB): a car drives past all 6 sign types plus 3 look-alike decoys (Speed Limit 20/70, turn left)
# in 6 conditions (noon, overcast, dusk, night with headlights, fog, dense fog). The simulator knows where each sign is, so every frame gives the true distance.
# Upload the generated `sim_results/` folder here (or put it on Drive). The CARLA scripts in `carla_demo/` write the same files and also work.
# Without such a folder this section is skipped and R6 stays NOT MEASURED, which is the honest answer.
#
# **What to check:** how the error grows with distance (far signs are only a few pixels tall), and which condition hurts detection most.
# Keep in mind the sim-to-real gap: rendered signs are cleaner than real photos, so treat simulation as *necessary* evidence, not *sufficient*.

# %%
SIM_CANDIDATES = [Path("/content/sim_results"), Path("/content/drive/MyDrive/traffic_sign_data/sim_results"), Path("webots_demo/sim_results"), Path("sim_results"),
                  Path("/content/carla_results"), WORK / "carla_results", Path("carla_results"), Path("/content/drive/MyDrive/carla_results")]
CARLA_DIR = next((p for p in SIM_CANDIDATES if (p / "distance_log.csv").exists()), None)
if CARLA_DIR is None:
    print("No simulation results found. Upload the sim_results (or carla_results) folder to /content and re-run this cell.")
else:
    dl, ws = (pd.read_csv(CARLA_DIR / f) if (CARLA_DIR / f).exists() else None for f in ("distance_log.csv", "weather_sweep.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    if dl is not None and len(dl):
        dl["abs_err_pct"] = (dl.est_dist_m - dl.gt_dist_m).abs() / dl.gt_dist_m * 100
        ax[0].scatter(dl.gt_dist_m, dl.est_dist_m, s=10, alpha=.5); lim = [0, max(dl.gt_dist_m.max(), dl.est_dist_m.max()) * 1.05]
        ax[0].plot(lim, lim, "k--", lw=1); ax[0].set_xlabel("ground-truth distance (m)"); ax[0].set_ylabel("estimated (m)"); ax[0].set_title(f"Pinhole distance vs simulator ground truth ({CARLA_DIR.name})")
        med = PACK["candidate"]["req"]["distance_err_pct"] = float(dl.abs_err_pct.median())
        print(f"median distance error {med:.1f}%  |  p90 {dl.abs_err_pct.quantile(.9):.1f}%  (R6 limit {REQS[5].target}%)")
    if ws is not None and len(ws):
        ax[1].bar(ws.weather, ws.detection_rate, color="tab:blue"); ax[1].set_ylim(0, 1); ax[1].set_ylabel("detection rate of visible signs")
        ax[1].tick_params(axis="x", rotation=60); ax[1].set_title("Detection rate of visible signs by condition (simulation)")
        display(ws.round(3))
    finish_plot("simulation")

# %% [markdown]
# # Final scorecard
# Everything measured in this notebook, against the requirements written in Step 1. **Anything that is not PASS is a work item, not a failure** — this is what an honest status report looks like.
# Read it as if you had to sign it: for each line, which section produced the evidence, and how much would you trust it?

# %%
SCORE = pd.DataFrame([{"req": r.rid, "requirement": r.text, "measured": v, "limit": f"{r.op} {r.target}{r.unit}",
                       "status": "NOT MEASURED" if ok is None else ("PASS" if ok else "FAIL")}
                      for r, v, ok in req_rows(PACK["candidate"]["req"])]).set_index("req")
display(SCORE)
GATE_DECISION, _ = release_gate(PACK["candidate"], PACK.get("v1"), verbose=False)
print(f"Release gate decision: {GATE_DECISION}")
print(f"Dataset {DATASET_ID} | near-duplicates between test and train: {LEAK_TEST_PCT:.0f}% | deployed variant: {PACK['candidate'].get('deploy_variant')}")
SCORE.to_csv(WORK / "scorecard.csv")

# %% [markdown]
# ## The loop
# ```
#   Requirements ──► Data ──► Training & validation ──► Gate ──► Deployment ──► Perception stack ──► Fleet / road
#        ▲                                                                                              │
#        └──────────── new requirements ◄── labelling queue ◄── drift alerts / shadow-mode disagreements ◄┘
# ```
# **Key ideas**
# 1. A requirement is testable; "good accuracy" is not.
# 2. The data decides the ceiling; audit it, slice it, version it, keep the test set honest.
# 3. Report by slice with uncertainty; gate on the worst case, not the average.
# 4. Speed is a requirement — and every optimisation must be re-validated.
# 5. A detector is a component; tracking, calibration, cross-checks and fallbacks make it a system.
# 6. After release, monitor, collect, retrain: the loop never closes.
#
# **Check yourself (Step 6)**
# 1. Why can't you monitor accuracy directly in the field, and what do you monitor instead?
# 2. Someone wants to promote a model whose gate said NO-SHIP "just for a test drive". What does the registry require, and why?
# 3. The final scorecard: which line is the weakest evidence, even if it says PASS?
#
# **Going further with your own car:** write your own requirements and ODD, collect and label images from *your* camera, run this slice table and gate automatically on every new model,
# calibrate the camera, and extend the Webots world in `webots_demo/` (curves, other cars, rain) for closed-loop tests. Good exercises in this notebook: retrain with `USE_DATASETS = ["a"]` and compare,
# change the cost table and watch the threshold move, or add a new degradation to the robustness sweep.
