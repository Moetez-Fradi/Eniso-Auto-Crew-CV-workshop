# %% [markdown]
# # Step 4 — Deployment constraints
# **What you will learn:** how to make a model fast enough for the car without silently breaking it.
#
# A model that is accurate but misses its **latency budget** is a failed model. Real-time means a *guaranteed worst case*, not a good average,
# and every optimisation must be **re-validated** — quantisation can quietly hurt the small and rare cases.
#
# **Background.** In a car the model runs on an embedded chip (an SoC or a small edge GPU) with strict power, heat and memory limits.
# The usual toolbox: a smaller model or input size, half precision (FP16), 8-bit integers (INT8), pruning, and inference compilers (ONNX Runtime, TensorRT, OpenVINO).
# Here R4 asks for p95 latency ≤ 66 ms, i.e. at least 15 frames per second for 95% of frames.
#
# The plan for this step: measure latency properly → export variants → evaluate each variant with the **same** code and threshold → pick the one that passes R1 *and* R4.
# Note: your timings depend on your GPU. A Colab T4 is slower than a recent laptop GPU, and the car's chip would be slower again.

# %% [markdown]
# ## 4.1 Measuring latency properly
# Latency is reported as **p50 / p95 / p99 / max**, not just the mean, on the full pipeline (pre-processing + inference + post-processing).
# p95 = 40 ms means 95% of frames finish within 40 ms; the remaining 5% are the ones that can make the car react late.

# %%
BENCH_IMGS = [cv2.imread(it["path"]) for it in ITEMS["test"][:40]]


def benchmark(fn, n=150, warmup=20):
    for i in range(warmup):
        fn(BENCH_IMGS[i % len(BENCH_IMGS)])
    ts = []
    for i in range(n):
        im = BENCH_IMGS[i % len(BENCH_IMGS)]
        t0 = time.perf_counter(); fn(im); ts.append((time.perf_counter() - t0) * 1000)
    p50, p95, p99 = np.percentile(ts, [50, 95, 99])
    return {"p50_ms": p50, "p95_ms": p95, "p99_ms": p99, "max_ms": np.max(ts), "fps_p50": 1000 / p50}


# %% [markdown]
# ## 4.2 A tiny ONNX inference path (letterbox → ORT → decode → NMS)
# Written by hand so exported models go through the same evaluation pipeline, and so you can see exactly what an embedded runtime has to do around the model itself.

# %%
import onnxruntime as ort


def letterbox(img, size, color=114):
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), color, np.uint8)
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, s, left, top


def to_input(lb):
    """letterboxed BGR uint8 image -> contiguous 1x3xHxW float32 RGB tensor in [0, 1]"""
    return np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0)


def _area(b):
    return (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])


def nms_per_class(boxes, scores, cls, iou_thr):
    """Greedy non-maximum suppression, applied separately for each class. Returns indices to keep."""
    order, keep = np.argsort(-scores), []
    while len(order):
        i, rest = order[0], order[1:]
        keep.append(int(i))
        inter = (np.clip(np.minimum(boxes[i, 2], boxes[rest, 2]) - np.maximum(boxes[i, 0], boxes[rest, 0]), 0, None)
                 * np.clip(np.minimum(boxes[i, 3], boxes[rest, 3]) - np.maximum(boxes[i, 1], boxes[rest, 1]), 0, None))
        iou_ = inter / np.maximum(_area(boxes[i]) + _area(boxes[rest]) - inter, 1e-9)
        order = rest[~((iou_ > iou_thr) & (cls[rest] == cls[i]))]
    return keep


class OrtDetector:
    def __init__(self, onnx_path, imgsz=IMGSZ, providers=None, label=None):
        providers = providers or (["CUDAExecutionProvider"] if HAS_GPU else []) + ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers)
        self.inp, self.imgsz, self.label = self.sess.get_inputs()[0].name, imgsz, label or Path(str(onnx_path)).stem
        self.provider = self.sess.get_providers()[0]
        self(np.zeros((416, 416, 3), np.uint8))

    def __call__(self, img, conf=0.01, iou=0.7):
        lb, s, left, top = letterbox(img, self.imgsz)
        out = self.sess.run(None, {self.inp: to_input(lb)})[0][0].T          # (anchors, 4 + nc)
        cls = out[:, 4:].argmax(1); confs = out[np.arange(len(out)), 4 + cls]
        sel = confs >= conf
        out, cls, confs = out[sel], cls[sel], confs[sel]
        if len(out) == 0:
            return []
        cx, cy, w, h = out[:, :4].T
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)               # xyxy in letterbox pixels
        order = np.argsort(-confs)[:1000]
        boxes, cls, confs = boxes[order], cls[order], confs[order]
        H, W = img.shape[:2]
        off, lim = (left, top), (W, H)                    # undo the letterbox, clip to the image
        return [{"box": tuple(float(np.clip((boxes[i, k] - off[k % 2]) / s, 0, lim[k % 2])) for k in range(4)),
                 "cls": int(cls[i]), "conf": float(confs[i])} for i in nms_per_class(boxes, confs, cls, iou)]


# %% [markdown]
# ## 4.3 Build the variants
# | Variant | Why it exists |
# |---|---|
# | PyTorch FP32 @640 | reference (what we validated) |
# | PyTorch FP32 @416 | cheaper input — but small signs have fewer pixels |
# | PyTorch FP16 @640 | half precision on GPU, usually free accuracy |
# | PyTorch FP32 on CPU | what a laptop / small board without GPU would do |
# | ONNX FP32 | portable graph for ONNX Runtime / TensorRT / OpenVINO toolchains |
# | ONNX INT8 | 4x smaller weights, fast on CPU/NPUs — **the risky one** |
# | TensorRT FP16 (optional) | vendor-optimised engine for NVIDIA edge devices |

# %%
VARIANTS = {"PyTorch FP32 @640": DET_MAIN, "PyTorch FP32 @416": YoloDetector(WEIGHTS_MAIN, 416, label="fp32_416")}
if HAS_GPU:
    VARIANTS["PyTorch FP16 @640"] = YoloDetector(WEIGHTS_MAIN, IMGSZ, half=True, label="fp16")
VARIANTS["PyTorch FP32 CPU @640"] = YoloDetector(WEIGHTS_MAIN, IMGSZ, device="cpu", label="cpu")

onnx_path = None
try:
    onnx_path = Path(YOLO(str(WEIGHTS_MAIN)).export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True, dynamic=False, device="cpu"))
    VARIANTS["ONNX FP32"] = OrtDetector(onnx_path, IMGSZ, label="onnx_fp32")
    print("ONNX exported:", onnx_path, f"({onnx_path.stat().st_size / 1e6:.1f} MB) | provider:", VARIANTS["ONNX FP32"].provider)
except Exception as e:
    print("ONNX export failed:", e)

# %%
# INT8 post-training static quantisation (needs a small calibration set of real images)
# per_channel=False: per-channel bias quantisation produces an invalid graph with onnx 1.23 + onnxruntime 1.31
if onnx_path is not None:
    try:
        import onnx
        from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

        class CalibReader(CalibrationDataReader):
            def __init__(self, paths, name, size):
                self.it, self.name, self.size = iter(paths), name, size

            def get_next(self):
                p = next(self.it, None)
                return None if p is None else {self.name: to_input(letterbox(cv2.imread(p), self.size)[0])}

        calib = [it["path"] for it in random.Random(SEED).sample(ITEMS["train"], 64)]
        in_name = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).get_inputs()[0].name
        head_idx = len(YOLO(str(WEIGHTS_MAIN)).model.model) - 1      # last module = detection head (22 in YOLOv8n, 23 in YOLO11)
        head = [n.name for n in onnx.load(str(onnx_path)).graph.node if n.name.startswith(f"/model.{head_idx}/")]   # keep the head in float
        int8_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
        quantize_static(str(onnx_path), str(int8_path), CalibReader(calib, in_name, IMGSZ), quant_format=QuantFormat.QDQ,
                        per_channel=False, weight_type=QuantType.QInt8, activation_type=QuantType.QInt8, nodes_to_exclude=head)
        VARIANTS["ONNX INT8 (CPU)"] = OrtDetector(int8_path, IMGSZ, providers=["CPUExecutionProvider"], label="onnx_int8")
        print(f"INT8 model: {int8_path.stat().st_size / 1e6:.1f} MB (FP32: {onnx_path.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print("INT8 quantisation skipped:", repr(e)[:300])

# %%
if TRY_TENSORRT and HAS_GPU:
    try:
        pip_install("tensorrt")
        eng = Path(YOLO(str(WEIGHTS_MAIN)).export(format="engine", half=True, imgsz=IMGSZ, device=0))
        VARIANTS["TensorRT FP16"] = YoloDetector(eng, IMGSZ, label="trt")
    except Exception as e:
        print("TensorRT skipped:", repr(e)[:300])

# %% [markdown]
# ## 4.4 Benchmark **and** re-validate every variant
# Same test images, same operating threshold, same metric code. If a variant is fast but loses Stop recall, it is not an optimisation — it is a regression.
#
# **What to check:** compare INT8 with FP32. How much faster is it on your hardware, and what did it cost in Stop recall and F1? Which variant would you put in the car?

# %%
rows, VARIANT_PACKS = [], {}
for name, det in VARIANTS.items():
    lat = benchmark(lambda im, d=det: d(im, OP_THR), n=40 if "CPU" in name or "INT8" in name else 150)   # realistic: run at the operating threshold
    pk = VARIANT_PACKS[name] = build_pack(det, name, [], run_detector(det, ITEMS["test"]), OP_THR, with_robustness=False)
    rows.append({"variant": name, **{k: round(v, 1) for k, v in lat.items()}, "F1": round(pk["metrics"]["f1"], 3),
                 "Stop recall": round(pk["req"]["stop_recall"], 3), "mAP50": round(pk["map50"], 3)})
bench = pd.DataFrame(rows).set_index("variant")
R1, R4 = REQS[0], REQS[3]
bench["R1 recall"] = ["PASS" if R1.check(v) else "FAIL" for v in bench["Stop recall"]]
bench["R4 latency"] = ["PASS" if R4.check(v) else "FAIL" for v in bench["p95_ms"]]
display(bench)

# %%
ax = bench[["p50_ms", "p95_ms", "p99_ms"]].plot(kind="barh", figsize=(9, 3.6), width=0.8)
ax.axvline(R4.target, color="r", ls="--"); ax.set_xlabel("latency (ms), red line = R4 budget"); ax.set_title("Latency percentiles per variant")
finish_plot("latency")

# %% [markdown]
# ## 4.5 Pick the deployable variant and re-run the gate
# Rule: among variants that **pass R1 and R4**, take the most accurate. If none passes, the variant closest to passing (best Stop recall) is carried forward and the gate stays closed;
# the next discussion is then about hardware, input size or training, not about shipping.
# The gate is re-run with the deployable variant, now with a measured R4. Only R6 (simulation, see section 6.4) should still be pending.

# %%
ok = bench[(bench["R1 recall"] == "PASS") & (bench["R4 latency"] == "PASS")]
if len(ok):
    DEPLOY_NAME = ok.sort_values(["F1", "p95_ms"], ascending=[False, True]).index[0]
    print("Deployable variant:", DEPLOY_NAME)
else:
    DEPLOY_NAME = bench.sort_values(["Stop recall", "F1", "p95_ms"], ascending=[False, False, True]).index[0]
    print("No variant passes both R1 and R4. Closest to passing:", DEPLOY_NAME, "-> the gate stays closed until the failing line is fixed.")

PACK["candidate"]["req"]["p95_latency_ms"] = float(bench.loc[DEPLOY_NAME, "p95_ms"])
if DEPLOY_NAME != "PyTorch FP32 @640":      # the shipped artifact is the optimised one: its accuracy numbers count
    for k in ("stop_recall", "other_min_recall", "bg_fa_per_100"):
        PACK["candidate"]["req"][k] = VARIANT_PACKS[DEPLOY_NAME]["req"][k]
PACK["candidate"]["deploy_variant"] = DEPLOY_NAME
GATE_DECISION, GATE_TABLE = release_gate(PACK["candidate"], PACK.get("v1"))

# %% [markdown]
# **Check yourself (Step 4)**
# 1. Two models have the same mean latency of 30 ms. One has p99 = 35 ms, the other p99 = 120 ms. Which one goes in the car, and why?
# 2. Why is re-validating after INT8 quantisation not optional? Which classes or slices would you expect to suffer first?
# 3. Reducing the input from 640 to 416 pixels is faster. Which requirement is most at risk, and which slice from Step 3 shows it?
