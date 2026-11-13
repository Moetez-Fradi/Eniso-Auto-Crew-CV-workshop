# %% [markdown]
# # Step 3 — Validation by scenario
# **What you will learn:** how to decide, with evidence, whether a model may replace the one in the car.
#
# One number is not evidence. In this step you (1) train or load the models, (2) pick the confidence threshold with **validation** data and the cost table,
# (3) evaluate the **test** set by slice (class, sign size, brightness, blur, degradations), (4) look at the failures one by one, and (5) put everything through a **release gate**.
# The test set is used once, at the end, and never to tune anything. If you tune on it, its score stops meaning anything.
#
# **Background.** Testing goes from cheap to expensive: offline dataset → replay of recorded drives → simulation → hardware-in-the-loop → real road.
# This notebook covers the offline level (and simulation in section 6.4). At every level, a new model must not be worse than the current one on any critical slice, even if its average is better: that is a **regression gate**.

# %% [markdown]
# ## 3.1 Detector wrapper and training
# All later steps (PyTorch, FP16, ONNX, INT8) use the same interface `detector(image) -> [{'box','cls','conf'}]`, so every variant is evaluated by the **same** code.
#
# Two models are prepared here: the **candidate** (YOLO11s on both datasets) and **v1** (the prototype, YOLOv8n on dataset a only), which plays the "model currently in the car".
# If you set `WEIGHTS_MAIN` / `WEIGHTS_V1` in the config, the weights are loaded; otherwise they are trained, which takes a long time.

# %%
from ultralytics import YOLO, settings
try:
    settings.update({"wandb": False})
except Exception:
    pass
WORKERS = 0 if os.name == "nt" else 2


class YoloDetector:
    def __init__(self, weights, imgsz=IMGSZ, half=False, device=DEVICE, label=None):
        self.model, self.weights, self.imgsz = YOLO(str(weights)), str(weights), imgsz
        self.half, self.device, self.label = bool(half and HAS_GPU), device, label or Path(str(weights)).stem
        self(np.zeros((416, 416, 3), np.uint8))               # warm-up

    def __call__(self, img, conf=0.01):
        b = self.model.predict(img, imgsz=self.imgsz, conf=conf, half=self.half, device=self.device, verbose=False)[0].boxes
        return [{"box": tuple(float(v) for v in xy), "cls": int(c), "conf": float(s)}
                for xy, c, s in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy())]


def train_model(base, epochs, imgsz, run_name, data=None):
    m = YOLO(base)
    m.train(data=str(data or DATA_YAML), epochs=epochs, imgsz=imgsz, batch=BATCH, patience=15, seed=SEED,
            project=str(WORK / "runs"), name=run_name, exist_ok=True, device=DEVICE, workers=WORKERS, plots=True,
            fliplr=0.0,                      # no horizontal flip: it would turn 'Keep Right' into 'Keep Left'
            degrees=5, scale=0.5, close_mosaic=min(10, max(1, epochs // 2)))
    return Path(m.trainer.best)


# %%
t0 = time.time()
if TRAIN_MAIN and not WEIGHTS_MAIN:
    WEIGHTS_MAIN = train_model(BASE_MODEL, EPOCHS, IMGSZ, "main")
assert WEIGHTS_MAIN, "Set WEIGHTS_MAIN or TRAIN_MAIN = True"
print("candidate weights:", WEIGHTS_MAIN, f"({time.time() - t0:.0f}s)")

# %%
# The "currently deployed" v1 = the original prototype: YOLOv8n, dataset 'a' only. The candidate must not regress against it.
if TRAIN_V1_BASELINE and not WEIGHTS_V1:
    WEIGHTS_V1 = train_model(V1_BASE, V1_EPOCHS, V1_IMGSZ, "v1", data=DATA_YAML_A if "a" in USE_DATASETS else DATA_YAML)
DET_MAIN = YoloDetector(WEIGHTS_MAIN, IMGSZ, label="candidate")
DET_V1 = YoloDetector(WEIGHTS_V1, V1_IMGSZ, label="v1") if WEIGHTS_V1 else None
print("v1:", WEIGHTS_V1)

# %% [markdown]
# ## 3.2 Cache predictions once, evaluate many times
# We run the detector at a very low confidence (0.01) and store everything. Any threshold, slice or metric afterwards is just arithmetic on that cache.

# %%
def run_detector(detector, items, transform=None, conf=0.01):
    recs = []
    for it in items:
        img = cv2.imread(it["path"])
        if transform is not None:
            img = transform(img)
        recs.append({"path": it["path"], "gts": it["gts"], "preds": detector(img, conf), "brightness": it["brightness"], "blur": it["blur"]})
    return recs


t0 = time.time()
VAL_MAIN, TEST_MAIN = run_detector(DET_MAIN, ITEMS["val"]), run_detector(DET_MAIN, ITEMS["test"])
print(f"cached {len(VAL_MAIN)} val + {len(TEST_MAIN)} test predictions in {time.time() - t0:.0f}s")

# Sanity check: our from-scratch mAP vs Ultralytics' official number (small differences are normal: NMS/IoU settings differ)
ap_own, map_own = dataset_ap50(TEST_MAIN)
official = YOLO(str(WEIGHTS_MAIN)).val(data=str(DATA_YAML), split="test", imgsz=IMGSZ, batch=BATCH, plots=False, verbose=False, device=DEVICE, workers=WORKERS)
print(f"mAP@0.5  ours = {map_own:.3f}   Ultralytics = {official.box.map50:.3f}")

# %% [markdown]
# ## 3.3 Choose the operating point with a cost model (on **validation**, not test)
# The confidence threshold is a safety decision. Lower → fewer missed signs, more phantom ones. The notebook picks the threshold that minimises the *priced* mistakes from Step 1.
#
# **What to check:** the cost curve and where its minimum is. Is the chosen threshold far from Ultralytics' default of 0.25? Is the curve flat or sharp around it (how much does a small change cost)?

# %%
THRS = np.round(np.arange(0.05, 0.96, 0.05), 2)


def choose_threshold(val_records):
    rows = []
    for t in THRS:
        cost, parts = expected_cost(val_records, t)
        m = metrics_from(per_image_counts(val_records, t))
        rows.append({"thr": t, "cost": cost, "precision": m["precision"], "recall": m["recall"], "stop_recall": m["cls_recall"][STOP], **parts})
    df = pd.DataFrame(rows).fillna(0)
    return float(df.loc[df.cost.idxmin(), "thr"]), df


OP_THR, sweep = choose_threshold(VAL_MAIN)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.3))
ax[0].plot(sweep.thr, sweep.precision, label="precision"); ax[0].plot(sweep.thr, sweep.recall, label="recall")
ax[0].plot(sweep.thr, sweep.stop_recall, label="Stop recall", ls="--"); ax[0].legend(); ax[0].set_title("Trade-off (validation set)")
cols = [c for c in COST if c in sweep]
ax[1].stackplot(sweep.thr, [sweep[c] for c in cols], labels=cols); ax[1].set_ylabel("cost"); ax[1].legend(fontsize=7)
ax[1].set_title(f"Priced mistakes -> chosen threshold = {OP_THR}")
for a in ax:
    a.axvline(OP_THR, color="r", ls=":"); a.set_xlabel("confidence threshold")
finish_plot("operating_point")

# %% [markdown]
# ## 3.4 Evaluate the test set — by slice
# `build_pack` computes everything the release gate needs for one model: headline metrics, per-class results, slices (class / size / brightness / blur / source) with **confidence intervals**, and robustness under degradations.
#
# **What to check:** the F1 interval printed under the headline numbers. The test set is small, so the interval is wide: two models whose intervals overlap a lot may not really differ.

# %%
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def degrade(img, kind, level):
    if kind == "brightness":
        return np.clip(img.astype(np.float32) * level, 0, 255).astype(np.uint8)
    if kind == "blur":
        k = int(level) * 2 + 1
        return cv2.GaussianBlur(img, (k, k), 0)
    if kind == "noise":
        return np.clip(img.astype(np.float32) + np.random.RandomState(SEED).normal(0, level, img.shape), 0, 255).astype(np.uint8)
    if kind == "scale":                                          # smaller = the sign looks further away
        small = cv2.resize(img, None, fx=level, fy=level, interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (img.shape[1], img.shape[0]))
    if kind == "jpeg":                                           # camera / ISP / transmission artefacts
        return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(level)])[1], cv2.IMREAD_COLOR)
    return img


# (kind, level, inside the ODD?)  -> only 'inside' conditions count for requirement R5
DEGRADATIONS = [("brightness", 0.3, False), ("brightness", 0.6, True), ("brightness", 1.4, True),
                ("blur", 1, True), ("blur", 2, True), ("blur", 3, False),
                ("noise", 10, True), ("noise", 20, False), ("noise", 30, False),
                ("scale", 0.75, True), ("scale", 0.5, True), ("scale", 0.35, False),
                ("jpeg", 30, True), ("jpeg", 10, False)]


def gt_table(records, thr):
    rows = []
    for r in records:
        m = match_image(r["preds"], r["gts"], thr)
        for j, g in enumerate(r["gts"]):
            w = g["box"][2] - g["box"][0]
            rows.append({"cls": g["cls"], "hit": m["gt_hit"][j], "size": band_size(w), "bright": band_brightness(r["brightness"]),
                         "blur": band_blur(r["blur"]), "source": SOURCE_NAMES.get(source_of(r["path"]), "?"), "path": r["path"], "w": w})
    return pd.DataFrame(rows)


def build_pack(detector, name, val_records, test_records, thr, with_robustness=True):
    pic = per_image_counts(test_records, thr)
    m = metrics_from(pic)
    no_gt = np.array([len(r["gts"]) == 0 for r in test_records])
    ap, mAP = dataset_ap50(test_records)
    g = gt_table(test_records, thr)
    other = [m["cls_recall"][c] for c in range(NC) if c != STOP and m["cls_n_gt"][c] > 0]
    pack = {"name": name, "detector": detector, "thr": thr, "val": val_records, "test": test_records,
            "pic": pic, "metrics": m, "ap": ap, "map50": mAP, "gt": g,
            "req": {"stop_recall": m["cls_recall"][STOP], "other_min_recall": min(other) if other else float("nan"),
                    "bg_fa_per_100": pic[no_gt][:, :, 1].sum() / max(no_gt.sum(), 1) * 100,
                    "worst_odd_f1": None, "p95_latency_ms": None, "distance_err_pct": None}}
    sl = {}    # slices: name -> dict(value, n, metric, ci)
    for key, label in [("cls", "class"), ("source", "source"), ("size", "size"), ("bright", "brightness"), ("blur", "blur")]:
        for v, sub in g.groupby(key):
            sl[f"{label}={TARGET_CLASSES[v] if key == 'cls' else v}"] = {"value": sub.hit.mean(), "n": len(sub), "metric": "recall",
                                                                         "ci": wilson(int(sub.hit.sum()), len(sub))}
    if with_robustness:
        rob_rows = []
        for kind, level, in_odd in DEGRADATIONS:
            recs = run_detector(detector, ITEMS["test"], transform=lambda im, k=kind, l=level: degrade(im, k, l))
            mm = metrics_from(per_image_counts(recs, thr))
            f1 = nan0(mm["f1"])
            rob_rows.append({"kind": kind, "level": level, "in_odd": in_odd, "f1": f1, "recall": mm["recall"], "precision": mm["precision"]})
            sl[f"degrade:{kind}={level}"] = {"value": f1, "n": len(recs), "metric": "F1", "ci": (float("nan"), float("nan"))}
        pack["robust"] = rob = pd.DataFrame(rob_rows)
        pack["req"]["worst_odd_f1"] = float(rob[rob.in_odd].f1.min())
    pack["slices"] = sl
    return pack


t0 = time.time()
PACK = {"candidate": build_pack(DET_MAIN, "candidate", VAL_MAIN, TEST_MAIN, OP_THR)}
if DET_V1:
    PACK["v1"] = build_pack(DET_V1, "v1", run_detector(DET_V1, ITEMS["val"]), run_detector(DET_V1, ITEMS["test"]), OP_THR)
print(f"evaluated in {time.time() - t0:.0f}s at threshold {OP_THR}")

# %%
P = PACK["candidate"]; M = P["metrics"]
print(f"TEST @ thr {OP_THR}:  precision {M['precision']:.3f} | recall {M['recall']:.3f} | F1 {M['f1']:.3f} | mAP@0.5 (own) {P['map50']:.3f}")
lo, hi = np.nanpercentile([metrics_from(P["pic"][np.random.RandomState(i).randint(0, len(P["pic"]), len(P["pic"]))])["f1"] for i in range(400)], [2.5, 97.5])
print(f"F1 95% bootstrap interval over test images: [{lo:.2f}, {hi:.2f}]  <- the test set is small, so the average alone is not trustworthy\n")
display(pd.DataFrame({"n_truth": M["cls_n_gt"], "precision": np.round(M["cls_precision"], 3), "recall": np.round(M["cls_recall"], 3),
                      "AP@0.5": [round(P["ap"].get(c, float("nan")), 3) for c in range(NC)]}, index=TARGET_CLASSES))

# %% [markdown]
# ### Slices — where does it break?
# Bars show **recall**; whiskers are 95% intervals. Slices with few samples (n < 20) are flagged: you cannot conclude anything from them.
# Find the weakest slice. Is it weak because the model is bad there, or because there are only a handful of examples?

# %%
sl = {k: v for k, v in P["slices"].items() if v["metric"] == "recall"}
vals, ci = np.array([v["value"] for v in sl.values()]), np.array([v["ci"] for v in sl.values()])
err = np.clip([vals - ci[:, 0], ci[:, 1] - vals], 0, None)   # clip float rounding at recall = 1.0
plt.figure(figsize=(11, 3.6))
plt.bar(range(len(sl)), vals, yerr=err, color=["tab:red" if v["n"] < 20 else "tab:blue" for v in sl.values()], capsize=3)
plt.xticks(range(len(sl)), [f"{k}\n(n={v['n']})" for k, v in sl.items()], rotation=60, ha="right", fontsize=7)
plt.axhline(REQS[0].target, color="k", ls=":", label="R1 target (Stop)"); plt.ylim(0, 1.05); plt.ylabel("recall"); plt.legend()
plt.title("Test recall by slice (red = fewer than 20 samples)"); finish_plot("slices")

# %% [markdown]
# ### Robustness under degradations
# Each test image is degraded on purpose (darker, blurrier, noisier, smaller, more compressed) and evaluated again, for both the candidate and v1.
# Only conditions **inside the ODD** count for requirement R5; the others show where the system stops being usable (and what monitoring in Step 5 and 6 must detect).

# %%
fig, ax = plt.subplots(1, 5, figsize=(16, 3))
for a, (kind, sub) in zip(ax, P["robust"].groupby("kind", sort=False)):
    a.bar([str(l) for l in sub.level], sub.f1, color=["tab:blue" if o else "lightgray" for o in sub.in_odd])
    a.axhline(M["f1"], color="k", ls=":"); a.axhline(REQS[4].target, color="r", ls="--"); a.set_ylim(0, 1); a.set_title(kind)
ax[0].set_ylabel("F1")
plt.suptitle("F1 under degradation (grey = outside the ODD, dotted = clean F1, red = R5 limit)", y=1.02, fontsize=9); finish_plot("robustness")
if "v1" in PACK:
    display(pd.DataFrame({k: PACK[k]["robust"].set_index(["kind", "level"]).f1 for k in PACK}).round(3))

# %% [markdown]
# ## 3.5 Failure mining
# Every miss and every false alarm gets a **reason**. The histogram of reasons is the to-do list for the next iteration:
# *small sign* → more resolution / more small examples, *dark* → night data, *wrong class* → look at 30 vs 50, and so on.
#
# **What to check:** look at the false alarms on background images in the grid. Are they all real model errors, or do some show a real sign that nobody labelled? (Section 2.6 found many of those.)

# %%
def mine_failures(records, thr):
    out = []
    for r in records:
        m = match_image(r["preds"], r["gts"], thr)
        wrong_gt = {j: i for i, j in m["wrong"].items()}
        kept = [r["preds"][i] for i in m["keep"]]
        for j, hit in enumerate(m["gt_hit"]):
            if hit:
                continue
            g = r["gts"][j]
            near_same = [p for p in r["preds"] if p["cls"] == g["cls"] and p["conf"] >= thr * 0.5 and 0.1 <= iou_xyxy(p["box"], g["box"]) < 0.5]
            if j in wrong_gt:
                reason = f"wrong class (read as {TARGET_CLASSES[r['preds'][wrong_gt[j]]['cls']]})"
            elif near_same:
                reason = "poor localisation"
            elif g["box"][2] - g["box"][0] < SIZE_EDGES[0]:
                reason = "small sign"
            elif band_brightness(r["brightness"]) == "dark":
                reason = "dark scene"
            elif band_blur(r["blur"]) == "blurry":
                reason = "blurry image"
            else:
                reason = "unexplained miss"
            out.append({"type": "MISS", "reason": reason, "path": r["path"], "gt": g, "preds": kept})
        out += [{"type": "FALSE ALARM", "reason": "on background image" if not r["gts"] else "extra / duplicate near a sign",
                 "path": r["path"], "gt": r["gts"][0] if r["gts"] else None, "preds": [r["preds"][i]]}
                for i in m["keep"] if not (m["pred_tp"][i] or i in m["wrong"])]
    return out


FAILS = mine_failures(TEST_MAIN, OP_THR)
cnt = Counter((f["type"], f["reason"].split(" (")[0]) for f in FAILS)
fdf = pd.DataFrame([{"type": t, "reason": r, "count": c} for (t, r), c in cnt.items()]).sort_values("count", ascending=False)
display(fdf.set_index(["type", "reason"]))
vis = []
for f in FAILS[:12]:
    im = cv2.imread(f["path"])
    if f["gt"] is not None:
        im = draw_boxes(im, [{"box": f["gt"]["box"], "label": "truth: " + TARGET_CLASSES[f["gt"]["cls"]]}], (0, 200, 0))
    vis.append(draw_boxes(im, [{"box": p["box"], "label": f"{TARGET_CLASSES[p['cls']]} {p['conf']:.2f}"} for p in f["preds"]], (0, 0, 255), 1))
show_grid(vis, [f"{f['type']}: {f['reason']}" for f in FAILS[:12]], cols=4, size=3.2)

# %% [markdown]
# ## 3.6 The release gate
# Inputs: the **requirements** (Step 1) and the **currently deployed model**. Output: SHIP / NO-SHIP / HOLD.
# * Every requirement must **PASS** (not measured yet → **PENDING**).
# * No slice may get worse than the deployed model by more than 5 points (if a slice has fewer than 20 samples → **REVIEW**, a human must look).
#
# At this point latency (R4) and distance error (R6) are not measured yet, so they show as PENDING and the gate cannot say SHIP. That is the gate doing its job:
# no evidence, no release. R4 is measured in Step 4 and the gate is re-run there.

# %%
def req_rows(req):
    """-> (requirement, measured value rounded for display, check result True/False/None) for every requirement"""
    for r in REQS:
        v = req.get(r.metric)
        yield r, None if v is None else round(float(v), 3), r.check(v)


def release_gate(cand, base=None, max_regress=0.05, min_n=20, verbose=True):
    rows = [{"gate": r.rid, "what": r.text[:58], "value": v, "limit": f"{r.op} {r.target}{r.unit}",
             "status": "PENDING" if ok is None else ("PASS" if ok else "FAIL")} for r, v, ok in req_rows(cand["req"])]
    if base is not None:
        for s, c in cand["slices"].items():
            b = base["slices"].get(s)
            if b is None or c["value"] != c["value"] or b["value"] != b["value"]:
                continue
            if b["value"] - c["value"] > max_regress:
                rows.append({"gate": "regression", "what": s, "value": round(c["value"], 3), "limit": f"deployed {b['value']:.3f} -0.05",
                             "status": "FAIL" if min(c["n"], b["n"]) >= min_n else "REVIEW"})
    df = pd.DataFrame(rows)
    st = set(df.status)
    decision = "NO-SHIP" if "FAIL" in st else ("HOLD (evidence missing / needs review)" if ({"PENDING", "REVIEW"} & st) else "SHIP")
    if verbose:
        display(df.set_index("gate"))
        print("DECISION:", decision)
    return decision, df


GATE_DECISION, GATE_TABLE = release_gate(PACK["candidate"], PACK.get("v1"))

# %% [markdown]
# **Exercise.** (1) Which single failing or pending line would you work on first, and how?
# (2) The cell below calls `release_gate(PACK['v1'], PACK['candidate'])`: it pretends the *old* model is the new candidate and the candidate is the one in the car. Read the output: this is what a regression looks like, and why the gate refuses it.
# (3) Would you change `max_regress` or `min_n`? What does each choice cost you?

# %%
# YOUR TURN: a candidate that is "better on average" but worse where it matters
_ = release_gate(PACK["v1"], PACK["candidate"]) if "v1" in PACK else print("train the v1 baseline to run this exercise")

# %% [markdown]
# **Check yourself (Step 3)**
# 1. Why must the threshold be chosen on validation and not on test? What would happen to the test score if you did it on test?
# 2. The candidate has a higher average than v1, but the gate would still block it if one slice got 6 points worse. Is that too strict? Give an example where it saves you.
# 3. A slice shows recall 1.00 with n = 4. What can you conclude?
# 4. The requirements file says R3 was relaxed from 2.0 to 4.0 false alarms per 100 background images. When is it acceptable to change a requirement after seeing results, and what should be written down when you do?
