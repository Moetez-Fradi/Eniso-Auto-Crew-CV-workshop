# %% [markdown]
# # Step 1 — Requirements and metrics
# **What you will learn:** how to turn a vague goal into requirements you can test, and how to build the metrics that test them.
#
# A goal like "detect Stop signs" cannot be tested: how often, how far away, in what light, how fast? A **requirement** can, because it fixes
# a target, the operating conditions, a threshold and a time budget. The operating conditions are called the **ODD** (operational design domain):
# the situations the system is designed for. Outside the ODD the system is not expected to work, but it must *notice* that it is outside.
#
# **Background: the three standards you will hear about**
# * **ISO 26262** (functional safety): what happens when hardware or software *fails*.
# * **ISO 21448 / SOTIF** (safety of the intended functionality): the system works exactly as designed, but the design is not good enough, e.g. the detector misses a sign in fog. Most perception problems live here.
# * **ISO/PAS 8800**: safety of AI components in road vehicles (data, training, validation of ML models).
#
# Below are the six requirements (R1–R6) this notebook uses. Every later step produces evidence for one of them, and the release gate in Step 3 checks them all.

# %%
@dataclass
class Requirement:
    rid: str
    text: str
    metric: str          # key looked up in the measured results
    op: str              # '>=' or '<='
    target: float
    unit: str = ""

    def check(self, value):
        """True / False, or None when the value was not measured."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return value >= self.target if self.op == ">=" else value <= self.target


ODD = ("Operational design domain (illustrative): daytime & dusk, dry or light rain, signs 10-40 m ahead, "
       "European-style signs, forward camera 640x480, urban/rural roads.")

REQS = [
    Requirement("R1", "Stop sign recall at the operating point",                    "stop_recall",           ">=", 0.95),
    Requirement("R2", "Recall of every other sign class",                           "other_min_recall",      ">=", 0.80),
    Requirement("R3", "False detections per 100 background images (no sign present)", "bg_fa_per_100",       "<=", 4.0),   # was 2.0; relaxed by the team (2026-11-13): a false sign alert is a nuisance, not a safety risk, at club speeds
    Requirement("R4", "p95 end-to-end latency",                                     "p95_latency_ms",        "<=", 66.0, "ms"),   # 15 FPS
    Requirement("R5", "Worst F1 over in-ODD degradations (blur, noise, brightness, scale, JPEG)", "worst_odd_f1", ">=", 0.60),
    Requirement("R6", "Distance estimation error (simulation: Webots or CARLA)",    "distance_err_pct",      "<=", 15.0, "%"),
]
print(ODD, "\n")
pd.DataFrame([vars(r) for r in REQS]).set_index("rid")

# %% [markdown]
# ## 1.1 Metric families
# Metrics come in families, from "does it find the box?" up to "is the car safe?". Rule of thumb: **mAP is a training and benchmark metric; safety is written in miss rate, false-alarm rate and latency.**

# %%
pd.DataFrame([
    ("Detection", "IoU, precision, recall, F1",              "Does it find the sign? How often is it wrong?"),
    ("Ranking",   "AP, mAP@0.5, mAP@0.5:0.95",              "Quality across all confidence thresholds / box tightness"),
    ("Per class", "confusion matrix, per-class AP",         "Which classes are confused (30 vs 50)?"),
    ("System",    "latency (p50/p95/p99), FPS, memory",     "Does it fit the hardware budget, *every* frame?"),
    ("Safety",    "miss rate by distance, false alarms/km", "What the requirement is actually written in"),
], columns=["family", "metrics", "answers"]).set_index("family")

# %% [markdown]
# ## 1.2 Metrics from scratch
# The evaluation is written by hand here (about 60 lines) and later checked against Ultralytics' own numbers, so nothing is a black box. Read the code once: it is short.
#
# * **IoU** = overlap / union of the predicted and true box.
# * A prediction is a **TP** if IoU ≥ 0.5 *and* the class is right (greedy matching, highest confidence first). Unmatched predictions = **FP**, unmatched truths = **FN**.

# %%
def iou_xyxy(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _best_gt(box, gts, skip, iou_thr):
    """Index of the not-skipped ground truth with the highest IoU >= iou_thr, or -1."""
    best, best_iou = -1, iou_thr - 1e-9
    for j, g in enumerate(gts):
        if not skip(j, g) and (v := iou_xyxy(box, g["box"])) > best_iou:
            best, best_iou = j, v
    return best


def match_image(preds, gts, conf_thr, iou_thr=0.5):
    """preds: [{'cls','conf','box'}], gts: [{'cls','box'}].
    Returns dict(keep=[pred idx], pred_tp={idx: bool}, gt_hit=[bool], wrong={pred idx: gt idx})
    'wrong' = unmatched predictions that overlap (IoU>=thr) an unmatched ground truth of ANOTHER class."""
    keep = sorted([i for i, p in enumerate(preds) if p["conf"] >= conf_thr], key=lambda i: -preds[i]["conf"])
    gt_used, pred_tp, wrong = [False] * len(gts), {}, {}
    for i in keep:
        best = _best_gt(preds[i]["box"], gts, lambda j, g: gt_used[j] or g["cls"] != preds[i]["cls"], iou_thr)
        pred_tp[i] = best >= 0
        if best >= 0:
            gt_used[best] = True
    for i in keep:
        if not pred_tp[i]:
            best = _best_gt(preds[i]["box"], gts, lambda j, g: gt_used[j] or j in wrong.values(), iou_thr)
            if best >= 0:
                wrong[i] = best
    return {"keep": keep, "pred_tp": pred_tp, "gt_hit": gt_used, "wrong": wrong}


def per_image_counts(records, thr, iou_thr=0.5):
    """-> array (N_images, NC, 3) with [tp, fp, fn] per image and class."""
    out = np.zeros((len(records), NC, 3))
    for k, r in enumerate(records):
        m = match_image(r["preds"], r["gts"], thr, iou_thr)
        for i in m["keep"]:
            out[k, r["preds"][i]["cls"], 0 if m["pred_tp"][i] else 1] += 1
        for j, hit in enumerate(m["gt_hit"]):
            if not hit:
                out[k, r["gts"][j]["cls"], 2] += 1
    return out


def metrics_from(pic, mask=None):
    """Aggregate a per_image_counts array (optionally only the images where mask is True)."""
    a = pic if mask is None else pic[np.asarray(mask, bool)]
    tp, fp, fn = a.sum(0).T
    TP, FP, FN = tp.sum(), fp.sum(), fn.sum()
    P = lambda t, f: t / (t + f) if (t + f) > 0 else float("nan")
    denom = 2 * TP + FP + FN
    return {"precision": P(TP, FP), "recall": P(TP, FN),
            "f1": 2 * TP / denom if denom > 0 else float("nan"),     # = harmonic mean of precision and recall
            "tp": int(TP), "fp": int(FP), "fn": int(FN),
            "cls_recall": [P(tp[c], fn[c]) for c in range(NC)], "cls_precision": [P(tp[c], fp[c]) for c in range(NC)],
            "cls_n_gt": [int(tp[c] + fn[c]) for c in range(NC)]}


def average_precision(conf, tp, n_gt):
    """101-point interpolated AP (COCO / Ultralytics style)."""
    if n_gt == 0 or len(conf) == 0:
        return float("nan"), np.array([]), np.array([])
    order = np.argsort(-np.asarray(conf))
    tp = np.asarray(tp, float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate([[1.0], prec, [0.0]]))))
    x = np.linspace(0, 1, 101)
    return float(_trapz(np.interp(x, mrec, mpre), x)), prec, rec


def dataset_ap50(records, iou_thr=0.5):
    """Per-class AP@0.5 and mAP@0.5 from cached predictions (records must hold low-confidence predictions)."""
    confs, flags, n_gt = defaultdict(list), defaultdict(list), Counter()
    for r in records:
        n_gt.update(g["cls"] for g in r["gts"])
        m = match_image(r["preds"], r["gts"], 0.0, iou_thr)
        for i in m["keep"]:
            c = r["preds"][i]["cls"]
            confs[c].append(r["preds"][i]["conf"]); flags[c].append(m["pred_tp"][i])
    ap = {c: average_precision(confs[c], flags[c], n_gt[c])[0] for c in range(NC) if n_gt[c] > 0}
    return ap, float(np.nanmean(list(ap.values()))) if ap else float("nan")

# %% [markdown]
# ### Toy example (no model needed): what IoU 0.5 looks like, and how precision/recall trade off
# **Try it:** the four panels shift the predicted box further from the truth. Before running, guess at which shift IoU drops below 0.5.
# Then change the shifts in `[0, 10, 20, 35]` and re-run. In the table below, follow how precision falls and recall rises as the threshold goes down.

# %%
gt_box = (100, 100, 160, 160)
fig, ax = plt.subplots(1, 4, figsize=(14, 3.2))
for a, shift in zip(ax, [0, 10, 20, 35]):
    pb = (100 + shift, 100 + shift // 2, 160 + shift, 160 + shift // 2)
    a.add_patch(plt.Rectangle(gt_box[:2], 60, 60, fill=False, ec="green", lw=2, label="truth"))
    a.add_patch(plt.Rectangle(pb[:2], 60, 60, fill=False, ec="red", lw=2, ls="--", label="prediction"))
    a.set_xlim(80, 220); a.set_ylim(220, 80); a.set_aspect("equal")
    v = iou_xyxy(gt_box, pb)
    a.set_title(f"IoU = {v:.2f}  ({'TP' if v >= 0.5 else 'miss'})")
ax[0].legend(fontsize=7); finish_plot()

# 8 toy predictions, sorted by confidence: lowering the threshold raises recall but admits false alarms
toy_conf = np.array([.95, .90, .85, .80, .70, .60, .45, .30])
toy_tp = np.array([1, 1, 1, 0, 1, 0, 1, 0])
toy_ngt = 6
ap, prec, rec = average_precision(toy_conf, toy_tp, toy_ngt)
plt.figure(figsize=(4.5, 3.5)); plt.plot(rec, prec, marker="o"); plt.xlabel("recall"); plt.ylabel("precision")
plt.title(f"Toy precision-recall curve, AP = {ap:.2f}"); plt.xlim(0, 1.02); plt.ylim(0, 1.05); finish_plot()
pd.DataFrame({"conf": toy_conf, "TP": toy_tp, "precision": prec.round(2), "recall": rec.round(2)})

# %% [markdown]
# ## 1.3 Errors are not equal — put a price on them
# The *same* detector can be "better" or "worse" depending on what each mistake costs. Missing a Stop sign is far worse than a phantom Keep Right.
# The cost table below is illustrative (arbitrary units; in a real project it comes from a hazard analysis). Step 3 uses it to **choose the confidence threshold**, instead of keeping 0.25 out of habit.
#
# **Try it:** after finishing Step 3, come back, double `miss_other`, and re-run 3.3. Which way does the threshold move, and why?

# %%
COST = {
    "miss_stop": 100,      # a missed Stop sign is a safety hazard
    "miss_other": 30,      # missed speed limit / No Entry / Give Way / Keep Right
    "wrong_class": 60,     # right place, wrong sign (e.g. 30 read as 50)
    "false_stop": 20,      # phantom Stop -> hard braking, rear-end risk
    "false_other": 5,      # phantom other sign -> mostly comfort
}


def expected_cost(records, thr, iou_thr=0.5):
    """Total cost of all mistakes made on `records` at confidence threshold `thr`."""
    parts = Counter()
    for r in records:
        m = match_image(r["preds"], r["gts"], thr, iou_thr)
        wrong_gt = set(m["wrong"].values())
        mistakes = ["wrong_class" if j in wrong_gt else "miss_stop" if r["gts"][j]["cls"] == STOP else "miss_other"
                    for j, hit in enumerate(m["gt_hit"]) if not hit]
        mistakes += ["false_stop" if r["preds"][i]["cls"] == STOP else "false_other"
                     for i in m["keep"] if not (m["pred_tp"][i] or i in m["wrong"])]
        for k in mistakes:
            parts[k] += COST[k]
    return sum(parts.values()), dict(parts)


pd.Series(COST, name="cost").to_frame()

# %% [markdown]
# **Check yourself (Step 1)**
# 1. Rewrite "the car should see speed limits" as a requirement with a target, conditions, a threshold and a time budget.
# 2. A detector has precision 0.99 and recall 0.60 on Stop signs. Is that good for a car? What about the reverse?
# 3. Why is R1 (Stop recall ≥ 0.95) stricter than R2 (other signs ≥ 0.80)? Find the answer in the cost table.
