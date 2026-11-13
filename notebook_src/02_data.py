# %% [markdown]
# # Step 2 — Data
# **What you will learn:** how to check a dataset before trusting any number computed on it.
#
# In production, teams improve the *data* more often than the model. Ask five questions of any dataset:
# is it **correct** (labels), does it **cover** the conditions in the ODD, is it **balanced**, is the test set **independent** of training, and can you **reproduce** it (version)?
#
# **Background.** Real driving data comes from a fleet of cars, public datasets and simulation. Labelling follows written guidelines and is audited,
# because annotators disagree and make mistakes. Data from video must be split **by drive**, not by frame, or the test set ends up full of near-copies of training images.
# This step does a small version of each of these checks.

# %% [markdown]
# ## 2.1 Get the datasets and select the 6 classes
# Two public sources: **a** = the prototype's original data (36 sign classes), **b** = *Street Sign Set* (7.3k European street images, 63 sign classes).
# Each file name gets its source as a prefix (`a_…`, `b_…`), so every result can later be split by source.
# Every sign that is not one of our 6 classes is dropped from the labels and becomes a **hard negative** (e.g. Speed Limit 40/60/70 teaches the model what is *not* a 30 or a 50).
# Images with none of our 6 classes stay in the data as **background (negative) images** — they are what lets us measure false alarms (requirement R3).
#
# **What to check in the output:** the table of images and boxes per source and split. Which source gives most of the training data? Which class is the rarest?
# The training list also repeats some images (*oversampling*, set in the config): the rare Keep Right class 3×, and signs that look like ours (curve warnings, Speed Limit 20/40) 2×.

# %%
def kaggle_auth():
    pip_install("--upgrade", "kaggle", "six")                      # the new KGAT_ tokens need a recent client
    kdir = Path.home() / ".kaggle"
    if ((kdir / "kaggle.json").exists() or (kdir / "access_token").exists()
            or "KAGGLE_USERNAME" in os.environ or "KAGGLE_API_TOKEN" in os.environ):
        return
    try:                                                # 1) Colab Secrets (key icon in the left bar): name = KAGGLE_API_TOKEN
        from google.colab import userdata
        os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
    except Exception:
        from getpass import getpass                     # 2) paste it: the input is hidden and never saved in the notebook
        tok = getpass("Paste your Kaggle token (starts with KGAT_), or leave empty to upload a kaggle.json: ").strip()
        if tok:
            os.environ["KAGGLE_API_TOKEN"] = tok
        else:                                           # 3) classic kaggle.json file
            from google.colab import files
            up = files.upload()
            kdir.mkdir(exist_ok=True)
            (kdir / "kaggle.json").write_bytes(next(iter(up.values())))
            (kdir / "kaggle.json").chmod(0o600)


def get_dataset_root(tag):
    ds = DATASETS[tag]
    if DATA_SOURCE == "local":
        return Path(ds["dir"] or DATA_ROOT_DIR / tag)
    if DATA_SOURCE == "drive":
        from google.colab import drive
        drive.mount("/content/drive")
        return DRIVE_DATA_DIR / tag
    dst = DATA_ROOT_DIR / tag; dst.mkdir(parents=True, exist_ok=True)
    if any(dst.rglob("*.yaml")):
        return dst
    kaggle_auth()
    r = subprocess.run(["kaggle", "datasets", "download", "-d", ds["kaggle"], "-p", str(dst), "--unzip"], capture_output=True, text=True)
    print(r.stdout[-500:], r.stderr[-500:])
    if r.returncode != 0:
        if not ds.get("hf"):
            raise RuntimeError(f"Kaggle download of {ds['kaggle']} failed, see the message above")
        print("Kaggle failed, trying Hugging Face (slow: ~15k small files, rate-limited)")
        pip_install("huggingface_hub")
        from huggingface_hub import snapshot_download
        snapshot_download(ds["hf"], repo_type="dataset", local_dir=str(dst), max_workers=8)
    return dst


def norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


ALIAS = {norm(a): TARGET_CLASSES.index(t) for t, al in CLASS_ALIASES.items() for a in al + [t]}


def read_source(root):
    """-> class names {id: name}, {split: folder with images/ and labels/} (works for the usual Kaggle / Roboflow layouts)"""
    ok = lambda p: ".cache" not in p.parts                          # skip download caches
    ymls = sorted([p for ext in ("*.yaml", "*.yml") for p in root.rglob(ext) if ok(p)],
                  key=lambda p: ("custom_data" not in p.name.lower(), str(p)))
    cfg = next(c for c in (yaml.safe_load(p.read_text(encoding="utf8")) for p in ymls) if isinstance(c, dict) and "names" in c)
    nm = cfg["names"]
    nm = dict(enumerate(nm)) if isinstance(nm, list) else {int(k): v for k, v in nm.items()}
    found = {sp: p for sp in ["train", "valid", "val", "test"]
             if (p := next((q for q in sorted(root.rglob(sp)) if ok(q) and q.is_dir() and (q / "images").is_dir() and (q / "labels").is_dir()), None))}
    return nm, {"train": found["train"], "val": found.get("valid") or found.get("val"), "test": found["test"]}


shutil.rmtree(OUT, ignore_errors=True)
SOURCE_STATS, ORIG_NAMES = [], {}
for tag in USE_DATASETS:
    names, splits = read_source(get_dataset_root(tag))
    src_to_new = {sid: ALIAS[norm(n)] for sid, n in names.items() if norm(n) in ALIAS}
    missing = set(range(NC)) - set(src_to_new.values())
    print(f"[{tag}] {DATASETS[tag]['name']}: {len(names)} classes;", {names[k]: TARGET_CLASSES[v] for k, v in src_to_new.items()},
          "| missing:", [TARGET_CLASSES[c] for c in missing] or "none")
    for split, sdir in splits.items():
        for d in ("images", "labels"):
            (OUT / d / split).mkdir(parents=True, exist_ok=True)
        n_img = n_box = 0
        for ip in sorted((sdir / "images").iterdir()):
            if ip.suffix.lower() not in IMG_EXT:
                continue
            kept, orig = [], set()
            lp = sdir / "labels" / (ip.stem + ".txt")
            for parts in map(str.split, lp.read_text().strip().splitlines() if lp.exists() else []):
                c = int(float(parts[0])) if parts else None
                if c in names:
                    orig.add(str(names[c]))
                if len(parts) >= 5 and c in src_to_new:
                    kept.append(f"{src_to_new[c]} " + " ".join(parts[1:5]))
            shutil.copy2(ip, OUT / "images" / split / f"{tag}_{ip.name}")
            (OUT / "labels" / split / f"{tag}_{ip.stem}.txt").write_text("\n".join(kept))
            n_img += 1; n_box += len(kept)
            if split == "train":
                ORIG_NAMES[f"{tag}_{ip.name}"] = orig
        SOURCE_STATS.append({"source": tag, "split": split, "images": n_img, "target boxes": n_box})

# Apply the label corrections (missing labels found by the review in 2.6)
LABEL_FIXES = None
fix_file = next((Path(c) for c in LABEL_FIXES_CANDIDATES if c and Path(c).exists()), None)
if fix_file:
    LABEL_FIXES = pd.read_csv(fix_file)
    LABEL_FIXES = LABEL_FIXES[LABEL_FIXES.file.str.split("_", n=1).str[0].isin(USE_DATASETS)]
    for r in LABEL_FIXES.itertuples():
        lp = OUT / "labels" / r.split / (Path(r.file).stem + ".txt")
        assert lp.exists(), f"label fix for unknown image {r.file}"
        if r.action == "add":
            old = lp.read_text().strip()
            lp.write_text((old + "\n" if old else "") + f"{TARGET_CLASSES.index(r.cls_name)} {r.x:.6f} {r.y:.6f} {r.w:.6f} {r.h:.6f}")
    print(f"label fixes applied from {fix_file}: {len(LABEL_FIXES)} boxes added", LABEL_FIXES.groupby("split").size().to_dict())
else:
    print("WARNING: label_fixes.csv not found -> using the raw labels (about 136 real signs are unlabelled, see 2.6)")

SOURCE_NAMES = {t: DATASETS[t]["name"].split(" (")[0] for t in USE_DATASETS}


def source_of(path):
    return Path(path).name.split("_", 1)[0]


# Training list with oversampling (repeating a file in the list = showing it more often per epoch)
train_list, rep_stats = [], Counter()
for ip in sorted((OUT / "images" / "train").iterdir()):
    lab = (OUT / "labels" / "train" / (ip.stem + ".txt")).read_text().split()
    cls_here = {TARGET_CLASSES[int(v)] for v in lab[0::5]}
    n = max([OVERSAMPLE_POS.get(c, 1) for c in cls_here] + [1])
    if not cls_here and ORIG_NAMES.get(ip.name, set()) & set(OVERSAMPLE_NEG[0]):
        n = OVERSAMPLE_NEG[1]; rep_stats["look-alike negatives"] += 1
    elif n > 1:
        rep_stats["rare positives"] += 1
    train_list += [str(ip)] * n
TRAIN_TXT = OUT / "train_list.txt"
TRAIN_TXT.write_text("\n".join(train_list))
print(f"training list: {len(train_list)} entries for {len(set(train_list))} images; repeated:", dict(rep_stats))


def write_data_yaml(name, train_txt):
    p = OUT / name
    p.write_text(yaml.safe_dump({"path": str(OUT), "train": str(train_txt), "val": "images/val", "test": "images/test",
                                 "names": dict(enumerate(TARGET_CLASSES))}, sort_keys=False))
    return p


DATA_YAML = write_data_yaml("data.yaml", TRAIN_TXT)
# training list for the v1 prototype: dataset 'a' only
TRAIN_A_TXT = OUT / "train_a.txt"
TRAIN_A_TXT.write_text("\n".join(str(p) for p in sorted((OUT / "images" / "train").glob("a_*"))))
DATA_YAML_A = write_data_yaml("data_a.yaml", TRAIN_A_TXT)
print("prepared:", OUT)
pd.DataFrame(SOURCE_STATS).pivot_table(index="source", columns="split", values=["images", "target boxes"]).astype(int)

# %% [markdown]
# ## 2.2 Build one table per image and per box
# For every image we measure what a *condition* looks like: **brightness**, **sharpness** (variance of the Laplacian), a **perceptual hash** (for leakage checks).
# These cheap statistics stand in for the "weather / time of day / blur" metadata that a real fleet dataset would carry.

# %%
def dhash(gray, size=8):
    s = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    return int("".join("1" if b else "0" for b in (s[:, 1:] > s[:, :-1]).flatten()), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


ITEMS, box_rows, img_rows = {}, [], []
for split in ["train", "val", "test"]:
    ITEMS[split] = []
    for ip in sorted((OUT / "images" / split).glob("*")):
        img = cv2.imread(str(ip)) if ip.suffix.lower() in IMG_EXT else None
        if img is None:
            continue
        H, W = img.shape[:2]
        gray, brightness, blur = quality(img)
        q = {"brightness": brightness, "blur": blur, "hash": dhash(gray)}
        gts, lp = [], OUT / "labels" / split / (ip.stem + ".txt")
        for line in (lp.read_text().strip() if lp.exists() else "").splitlines():
            c, x, y, w, h = line.split()[:5]
            c, x, y, w, h = int(c), float(x), float(y), float(w), float(h)
            gts.append({"cls": c, "box": yolo_to_xyxy(x, y, w, h, W, H)})
            box_rows.append({"split": split, "path": str(ip), "cls": c, "cx": x, "cy": y, "w": w, "h": h, "w_px": w * W, "h_px": h * H})
        ITEMS[split].append({"path": str(ip), "W": W, "H": H, "gts": gts, **q, "source": source_of(ip)})
        img_rows.append({"split": split, "path": str(ip), "source": source_of(ip), "n_boxes": len(gts), **q})
IMGS, BOXES = pd.DataFrame(img_rows), pd.DataFrame(box_rows)
BOXES["cls_name"] = BOXES["cls"].map(dict(enumerate(TARGET_CLASSES)))

summary = BOXES.pivot_table(index="cls_name", columns="split", values="cx", aggfunc="count").reindex(TARGET_CLASSES)[["train", "val", "test"]]
summary.loc["(background images)"] = IMGS[IMGS.n_boxes == 0].groupby("split").size()
summary.loc["(all images)"] = IMGS.groupby("split").size()
summary

# %% [markdown]
# ### Sample entries
# Before computing anything else, look at the data itself: a few rows of each table, one raw label file, and one training image per class and source with its ground-truth boxes.

# %%
display(IMGS.sample(5, random_state=SEED))
display(BOXES.sample(5, random_state=SEED)[["split", "path", "cls_name", "cx", "cy", "w", "h", "w_px", "h_px"]])

# one random example per class and source, where that sign is clearly visible (30 px wide up to 60% of the image width)
rs = random.Random(SEED)
samples = []
for c in range(NC):
    for t in USE_DATASETS:
        cand = [it for it in ITEMS["train"] if it["source"] == t
                and any(g["cls"] == c and 30 <= g["box"][2] - g["box"][0] <= 0.6 * it["W"] for g in it["gts"])]
        if cand:
            samples.append((c, rs.choice(cand)))
lp = OUT / "labels" / "train" / (Path(samples[0][1]["path"]).stem + ".txt")
print(f"raw label file {lp.name}  (class cx cy w h, normalised to [0, 1]):\n{lp.read_text()}")
show_grid([draw_boxes(cv2.imread(it["path"]), [{"box": g["box"], "label": TARGET_CLASSES[g["cls"]]} for g in it["gts"]]) for _, it in samples],
          [f"{TARGET_CLASSES[c]} | {SOURCE_NAMES[it['source']]}" for c, it in samples], cols=4)

# %% [markdown]
# ## 2.3 Define the condition bands (from the **training** set only)
# The brightness and blur scales are cut at fixed thresholds taken from train, then the *same* thresholds are reused for validation, test and — in Step 6 — field data.
# If each split had its own thresholds, "dark" would mean something different in each one and the comparisons would be meaningless.

# %%
BR_LO, BR_HI = np.percentile(IMGS[IMGS.split == "train"].brightness, [20, 80])
BL_LO = np.percentile(IMGS[IMGS.split == "train"].blur, 20)
SIZE_EDGES = (16, 32)       # sign width in pixels: small < 16 <= medium < 32 <= large


def band_brightness(v):
    return "dark" if v < BR_LO else ("bright" if v > BR_HI else "normal")


def band_blur(v):
    return "blurry" if v < BL_LO else "sharp"


def band_size(w):
    return "small" if w < SIZE_EDGES[0] else ("medium" if w < SIZE_EDGES[1] else "large")


IMGS["bright_band"] = IMGS.brightness.map(band_brightness)
IMGS["blur_band"] = IMGS.blur.map(band_blur)
BOXES["size_band"] = BOXES.w_px.map(band_size)
BOXES = BOXES.merge(IMGS[["path", "brightness", "blur", "bright_band", "blur_band"]], on="path", how="left")
print(f"dark < {BR_LO:.0f} < normal < {BR_HI:.0f} < bright | blurry: Laplacian var < {BL_LO:.0f} | small < {SIZE_EDGES[0]}px <= medium < {SIZE_EDGES[1]}px <= large")

# %% [markdown]
# ## 2.4 Coverage grid — where are the holes?
# A big dataset can still be empty in the cells that matter. Count boxes per (class × size) and per (brightness × size) in **train** and **test**.
#
# **What to check:** find the cells with very few boxes (for example small signs in the dark). Any result the model gets on those cells later is based on almost no evidence.

# %%
fig, ax = plt.subplots(2, 2, figsize=(11, 7))
for r, split in enumerate(["train", "test"]):
    b = BOXES[BOXES.split == split]
    for c, (key, title, order) in enumerate([("cls_name", "class", TARGET_CLASSES), ("bright_band", "brightness", ["dark", "normal", "bright"])]):
        t = pd.crosstab(b[key], b["size_band"]).reindex(index=order, columns=["small", "medium", "large"], fill_value=0)
        a = ax[r, c]; a.imshow(t.values, cmap="YlGnBu")
        a.set_xticks(range(3)); a.set_xticklabels(t.columns); a.set_yticks(range(len(t))); a.set_yticklabels(t.index, fontsize=8)
        for (i, j), v in np.ndenumerate(t.values):
            a.text(j, i, int(v), ha="center", va="center", color="red" if v < 10 else "black", fontsize=9)
        a.set_title(f"{split}: boxes by {title} x sign size  (red = fewer than 10)", fontsize=9)
finish_plot("coverage_grid")
print(f"Median sign width {BOXES.w_px.median():.0f}px; {(BOXES.w_px < SIZE_EDGES[0]).mean() * 100:.0f}% of signs are 'small' (<{SIZE_EDGES[0]}px).")

# %% [markdown]
# ## 2.5 Class imbalance
# Imbalance decides *how you report* (per class, never one average) and what you collect next.

# %%
tr = BOXES[BOXES.split == "train"].cls_name.value_counts().reindex(TARGET_CLASSES)
plt.figure(figsize=(7, 3)); plt.bar(tr.index, tr.values, color="tab:blue")
plt.xticks(rotation=25, ha="right"); plt.ylabel("train boxes"); plt.title(f"Imbalance: {tr.max() / tr.min():.1f}x between largest and smallest class"); finish_plot()

# %% [markdown]
# ## 2.6 Label audit
# Bad labels cap the accuracy you can ever reach, and they silently corrupt your metrics. Automatic checks catch the cheap errors; humans review the rest.

# %%
def draw_yolo_box(path, x, y, w, h, label):
    im = cv2.imread(str(path)); H, W = im.shape[:2]
    return draw_boxes(im, [{"box": yolo_to_xyxy(x, y, w, h, W, H), "label": label}], (0, 0, 255))


b = BOXES
tiny = (b.w_px < 4) | (b.h_px < 4)
extreme = (b.w_px / b.h_px.clip(lower=1e-6) > 2.5) | (b.h_px / b.w_px.clip(lower=1e-6) > 2.5)
audit = pd.DataFrame([
    ("box outside image (centre/size not in [0,1])", int(((b.cx < 0) | (b.cx > 1) | (b.cy < 0) | (b.cy > 1) | (b.w <= 0) | (b.h <= 0) | (b.w > 1) | (b.h > 1)).sum())),
    ("tiny box (< 4 px wide or high)", int(tiny.sum())),
    ("extreme aspect ratio (> 2.5 : 1)", int(extreme.sum())),
    ("duplicate boxes (same class, IoU > 0.9)", sum(g1["cls"] == g2["cls"] and iou_xyxy(g1["box"], g2["box"]) > 0.9
                                                    for its in ITEMS.values() for it in its for g1, g2 in combinations(it["gts"], 2))),
    ("images without label file", int(sum(not (OUT / "labels" / Path(p).parent.name / (Path(p).stem + ".txt")).exists() for p in IMGS.path))),
], columns=["check", "count"]).set_index("check")
display(audit)
odd = b[tiny | extreme].head(8)
if len(odd):
    show_grid([draw_yolo_box(r.path, r.cx, r.cy, r.w, r.h, r.cls_name) for r in odd.itertuples()], [Path(p).name[:22] for p in odd.path], cols=4)

# %% [markdown]
# **Look at the crops, per class and per source.** Automatic checks do not catch *semantic* label errors.
# In source **a** the class originally named *Turn Right* is mostly the blue **keep right** sign, mixed with real turn-right arrows and even a curve-warning triangle:
# one class name, several different signs. That is why we call the class *Keep Right* and map source b's `mand_pass_right` to it.

# %%
def class_crops(cls, source, n=10, size=72):
    out = []
    for it in ITEMS["train"]:
        if it["source"] != source:
            continue
        for g in it["gts"]:
            if g["cls"] == cls:
                x1, y1, x2, y2 = map(int, g["box"])
                cr = cv2.imread(it["path"])[max(0, y1):y2, max(0, x1):x2]
                if cr.size:
                    out.append(cv2.resize(cr, (size, size)))
        if len(out) >= n:
            break
    out = out[:n]
    return np.hstack(out + [np.full((size, size, 3), 255, np.uint8)] * (n - len(out)))


rows = [(f"{c} | {SOURCE_NAMES[t]}", class_crops(TARGET_CLASSES.index(c), t)) for c in ["Keep Right", "Give Way"] for t in USE_DATASETS]
plt.figure(figsize=(12, 1.4 * len(rows)))
for i, (title, strip) in enumerate(rows):
    plt.subplot(len(rows), 1, i + 1); plt.imshow(cv2.cvtColor(strip, cv2.COLOR_BGR2RGB)); plt.axis("off"); plt.title(title, fontsize=9, loc="left")
finish_plot("label_crops")

# %% [markdown]
# **Model-assisted label review.** A trained model was run over *all* splits, and every confident detection (conf ≥ 0.5) that overlaps no label was listed
# (the script is `experiments/mine_missing_labels.py`). A person looked at each of the 230 candidates: 136 were real signs nobody had labelled (67 of them Give Way!),
# the other 94 were genuine model errors (curve warnings read as *Keep Right*, Speed Limit 20/70/90, road markings).
# The accepted ones live in `label_fixes.csv`, a reviewed and versioned correction layer applied in 2.1 — the raw data is never edited.
#
# Missing labels hurt twice: they teach the model to *ignore* real signs, and they count correct detections as *false alarms* (R3).
# Keep this in mind when comparing models: part of any improvement measured on corrected labels comes from the labels, not the model.

# %%
if LABEL_FIXES is not None:
    display(LABEL_FIXES.pivot_table(index="cls_name", columns="split", values="file", aggfunc="count", fill_value=0))
    sample = LABEL_FIXES.sample(min(8, len(LABEL_FIXES)), random_state=SEED)
    show_grid([draw_yolo_box(OUT / "images" / r.split / r.file, r.x, r.y, r.w, r.h, "added: " + r.cls_name) for r in sample.itertuples()],
              [f"{r.split} | {r.file[:20]}" for r in sample.itertuples()], cols=4)

# %% [markdown]
# ## 2.7 Leakage check — is the test set really *unseen*?
# Consecutive video frames (or augmented copies) look almost identical. If a near-duplicate of a test image is in train, the test score is **optimistic**.
# We use a perceptual hash (dHash); a Hamming distance ≤ 4 out of 64 bits means "visually the same image".
# *Fix in real projects:* split by **drive / location**, never by random frame.

# %%
tr_h = [it["hash"] for it in ITEMS["train"]]
leak = pd.DataFrame([{"split": s, "path": it["path"], "min_hamming": min(hamming(it["hash"], h) for h in tr_h)}
                     for s in ["val", "test"] for it in ITEMS[s]])
plt.figure(figsize=(6, 3))
for s, c in [("val", "tab:orange"), ("test", "tab:green")]:
    plt.hist(leak[leak.split == s].min_hamming, bins=range(0, 33), alpha=.6, label=s, color=c)
plt.axvline(4.5, color="r", ls="--"); plt.xlabel("distance to nearest TRAIN image (bits)"); plt.legend(); plt.title("Near-duplicate check"); finish_plot()
leak["dup"], leak["source"] = leak.min_hamming <= 4, leak.path.map(source_of)
for s in ["val", "test"]:
    n, tot = int(leak[leak.split == s].dup.sum()), int((leak.split == s).sum())
    print(f"{s}: {n}/{tot} images ({n / tot * 100:.0f}%) have a near-duplicate in train")
leak_test = leak[leak.split == "test"]
print("\nshare of test images with a near-duplicate in train, per source:")
print((leak_test.groupby("source").dup.mean() * 100).round(0).rename(SOURCE_NAMES).to_string())
LEAK_TEST_PCT = float(leak_test.dup.mean() * 100)
if LEAK_TEST_PCT > 5:
    print("\n>>> Warning: a noticeable share of the test set is almost identical to training images; test scores below are likely optimistic.")

# %% [markdown]
# ## 2.8 Rare-event mining
# With a fleet you cannot label everything. You *rank* images by how informative they are and label the top of the list first.
# Here: rare class + small sign + dark scene = high priority. Look at the top images: would you agree they are the most useful ones to label?

# %%
freq = tr / tr.sum()


def label_priority(it):
    return (sum(1.0 / freq[TARGET_CLASSES[g["cls"]]] for g in it["gts"])
            + 5 * sum((g["box"][2] - g["box"][0]) < SIZE_EDGES[0] for g in it["gts"])
            + 10 * (it["brightness"] < BR_LO))


top = sorted(((label_priority(it), it) for it in ITEMS["train"] if it["gts"]), key=lambda t: -t[0])[:8]
show_grid([draw_boxes(cv2.imread(it["path"]), [{"box": g["box"], "label": TARGET_CLASSES[g["cls"]]} for g in it["gts"]], (0, 200, 255)) for _, it in top],
          [f"priority {s:.0f}" for s, _ in top], cols=4)

# %% [markdown]
# ## 2.9 Version the dataset
# Every result must be traceable to the exact data that produced it. We hash all images + labels into one **dataset ID** and write a manifest.

# %%
def dataset_version():
    h, n_files = hashlib.sha256(), 0
    for split in ["train", "val", "test"]:
        for ip in sorted((OUT / "images" / split).glob("*")):
            lp = OUT / "labels" / split / (ip.stem + ".txt")
            h.update(ip.name.encode()); h.update(ip.read_bytes()); h.update(lp.read_bytes() if lp.exists() else b"")
            n_files += 1
    return h.hexdigest()[:10], n_files


DATASET_ID, n_files = dataset_version()
manifest = {"dataset_id": DATASET_ID, "n_images": n_files, "classes": TARGET_CLASSES, "sources": {t: DATASETS[t] for t in USE_DATASETS},
            "split_counts": summary.loc[TARGET_CLASSES].to_dict(), "near_duplicate_test_pct": round(LEAK_TEST_PCT, 1),
            "condition_bands": {"brightness_lo": float(BR_LO), "brightness_hi": float(BR_HI), "blur_lo": float(BL_LO)}}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
print("dataset id:", DATASET_ID, "| images:", n_files)

# %% [markdown]
# **Check yourself (Step 2)**
# 1. Which class is the rarest in the test set? How many test boxes does it have, and how far would recall move if the model missed two more of them?
# 2. Why do we keep images that contain *other* signs (Speed Limit 40, curve warnings) instead of deleting them?
# 3. The leakage check found some near-duplicates between train and test. What does that do to the test score, and how would you split the data to avoid it?
# 4. You add 50 new images and fix 3 labels. Does the dataset ID change? Why does that matter when you compare two models?
