"""Webots controller: drive past the signs under several lighting conditions, run the detector on every frame,
and score it against the simulator's ground truth (true sign position -> true distance and true box).

Writes (same format as the CARLA scripts, read by notebook section 6.4):
  distance_log.csv   one row per (frame, visible sign) that was detected with the right class: true vs estimated distance
  observations.csv   one row per (frame, visible sign): detected or not, predicted class
  weather_sweep.csv  one row per condition: detection rate, sign recall, wrong-class rate, false alarms, distance error
"""
import csv, json, math, sys, time
from pathlib import Path

import numpy as np
import cv2
from controller import Supervisor, Display

CFG = json.loads((Path(__file__).with_name("run_config.json")).read_text())
OUT = Path(CFG["out"]); OUT.mkdir(parents=True, exist_ok=True)

# lighting presets: sky luminosity, sun, ambient, fog visibility (0 = no fog), headlight
CONDITIONS = {
    "noon":      dict(sky=1.0,  sun=1.6,  amb=0.5,  fog=0,  head=False, sky_col=(0.62, 0.76, 0.92), fog_col=(0.75, 0.78, 0.8)),
    "overcast":  dict(sky=0.7,  sun=0.25, amb=0.6,  fog=0,  head=False, sky_col=(0.7, 0.72, 0.75),  fog_col=(0.75, 0.78, 0.8)),
    "dusk":      dict(sky=0.35, sun=0.15, amb=0.2,  fog=0,  head=True,  sky_col=(0.85, 0.55, 0.4),  fog_col=(0.5, 0.4, 0.35)),
    "night":     dict(sky=0.03, sun=0.0,  amb=0.03, fog=0,  head=True,  sky_col=(0.05, 0.06, 0.1),  fog_col=(0.05, 0.05, 0.06)),
    "fog":       dict(sky=0.8,  sun=0.6,  amb=0.5,  fog=70, head=False, sky_col=(0.75, 0.78, 0.8),  fog_col=(0.75, 0.78, 0.8)),
    "dense_fog": dict(sky=0.7,  sun=0.4,  amb=0.5,  fog=35, head=True,  sky_col=(0.72, 0.74, 0.76), fog_col=(0.72, 0.74, 0.76)),
}
MIN_D, MAX_D = 5.0, 40.0        # distances scored (same as the CARLA scripts)

sup = Supervisor()
dt = int(sup.getBasicTimeStep())
cam = sup.getDevice("camera"); cam.enable(dt)
W, H = cam.getWidth(), cam.getHeight()
F = (W / 2) / math.tan(cam.getFov() / 2)              # focal length in pixels (Webots fov is horizontal)
DISPLAY = sup.getDevice("detector_view") if CFG.get("show") else None   # Display node shown as an overlay in the 3D view
DW, DH = (DISPLAY.getWidth(), DISPLAY.getHeight()) if DISPLAY is not None else (640, 360)
WIN = "Detector view (what the model sees)"
car = sup.getSelf(); car_tr = car.getField("translation")
x0, lane_y, cam_z = car_tr.getSFVec3f()
node = lambda d: sup.getFromDef(d)
SKY, SUN, AMB, FOG, HEAD = node("SKY"), node("SUN"), node("AMBIENT"), node("FOG"), node("CAR.HEADLIGHT")


def set_condition(c):
    SKY.getField("luminosity").setSFFloat(c["sky"]); SKY.getField("skyColor").setMFColor(0, list(c["sky_col"]))
    SUN.getField("intensity").setSFFloat(c["sun"]); AMB.getField("intensity").setSFFloat(c["amb"])
    FOG.getField("visibilityRange").setSFFloat(float(c["fog"])); FOG.getField("color").setSFColor(list(c["fog_col"]))
    if HEAD:
        HEAD.getField("on").setSFBool(c["head"])


# ground-truth sign centres (world frame) from the panel solids
SIGNS = []
for s in CFG["signs"]:
    sn = node(f"SIGN_{s['id']}")
    panel = sn.getField("signBoards").getMFNode(0)
    sup.step(dt)
    p = np.array(panel.getPosition()); p[2] += s["dz"]
    SIGNS.append({**s, "pos": p})
print("sign centres:", [(s["family"], np.round(s["pos"], 2).tolist()) for s in SIGNS], flush=True)

for _ in range(150):           # warm-up: sign textures are fetched/loaded asynchronously on first use
    sup.step(dt)

from ultralytics import YOLO   # imported after Webots is up so start-up errors are visible early
model = YOLO(CFG["weights"])
NAMES = model.names
model.predict(np.zeros((H, W, 3), np.uint8), imgsz=CFG["imgsz"], verbose=False)


def gt_box(s, cam_pos):
    """Project the sign panel into the image. Camera looks along +x (world), y = left, z = up."""
    rel = s["pos"] - cam_pos
    d = rel[0]                                     # depth along the optical axis = what the pinhole formula estimates
    if d <= 1.0:
        return None, d
    u, v = W / 2 - F * rel[1] / d, H / 2 - F * rel[2] / d
    hw, hh = F * s["w"] / 2 / d, F * s["h"] / 2 / d
    return (u - hw, v - hh, u + hw, v + hh), d


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def write_csv(path, rows):
    if not rows:
        Path(path).write_text(""); return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def draw(frame, cond, dets, vis, k, fps):
    for g, box, d, det in vis:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0) if g["family"] else (160, 160, 160), 1)
    for dd in dets:
        x1, y1, x2, y2 = map(int, dd["box"])
        col = (0, 0, 255) if dd.get("false") else (255, 160, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        txt = f"{dd['name']} {dd['conf']:.2f}"
        if dd.get("est") is not None:
            txt += f" | est {dd['est']:.1f} m / true {dd['true']:.1f} m"
        cv2.putText(frame, txt, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (W, 34), (0, 0, 0), -1)
    cv2.putText(frame, f"Webots | condition: {cond} | frame {k} | detector {fps:.0f} FPS | green = true sign, orange = detection, red = false alarm",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


dist_rows, obs_rows, sweep = [], [], []
writer = None
end_x = CFG["first_x"] + CFG["spacing"] * (len(SIGNS) - 1) + 3
t_start = time.time()
for cname in CFG["conditions"]:
    c = CONDITIONS[cname]; set_condition(c)
    record = bool(CFG["video"]) and cname in CFG["video_conditions"]
    if record and writer is None:
        writer = cv2.VideoWriter(str(OUT / CFG["video"]), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))   # 0.5 m/frame at 30 fps = 54 km/h
    x, k, false_alarms, lat = x0, 0, 0, []
    car_tr.setSFVec3f([x, lane_y, cam_z]); sup.step(dt); sup.step(dt)
    while x < end_x:
        x += CFG["step_m"]; car_tr.setSFVec3f([x, lane_y, cam_z])
        if sup.step(dt) == -1:
            sys.exit(0)
        img = np.frombuffer(cam.getImage(), np.uint8).reshape(H, W, 4)[:, :, :3].copy()
        cam_pos = np.array([x, lane_y, cam_z])
        t0 = time.perf_counter()
        r = model.predict(img, imgsz=CFG["imgsz"], conf=CFG["conf"], verbose=False)[0]
        lat.append(time.perf_counter() - t0)
        dets = [dict(box=b.tolist(), cls=int(cl), name=NAMES[int(cl)], conf=float(cf))
                for b, cl, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy())]
        # every sign ahead, at any distance (a detection on a far real sign is not a false alarm)
        ahead = [(g, *gt_box(g, cam_pos)) for g in SIGNS]
        ahead = [(g, b, d) for g, b, d in ahead if b is not None and b[2] > 0 and b[0] < W and b[3] > 0 and b[1] < H]
        used = set()
        vis = []
        for g, b, d in ahead:
            if not (MIN_D <= d <= MAX_D and b[0] >= 0 and b[2] <= W and b[1] >= 0 and b[3] <= H):
                continue
            best = max(range(len(dets)), key=lambda j: iou(b, dets[j]["box"]), default=None)
            hit = dets[best] if best is not None and iou(b, dets[best]["box"]) >= 0.3 else None
            if hit is not None:
                used.add(best)
            vis.append((g, b, d, hit))
            if g["family"] is None:          # decoy sign: any detection on it is a false alarm (counted below)
                if hit is not None:
                    used.discard(best)
                continue
            ok = hit is not None and hit["name"] == g["family"]
            est = F * g["h"] / max(hit["box"][3] - hit["box"][1], 1) if ok else None
            obs_rows.append({"frame": k, "weather": cname, "sign_id": g["id"], "family": g["family"], "gt_dist_m": round(d, 2),
                             "detected": ok, "pred_class": hit["name"] if hit else "", "conf": round(hit["conf"], 3) if hit else ""})
            if ok:
                hit["est"], hit["true"] = est, d
                dist_rows.append({"frame": k, "weather": cname, "sign_id": g["id"], "family": g["family"], "conf": round(hit["conf"], 3),
                                  "gt_dist_m": round(d, 2), "est_dist_m": round(est, 2), "err_pct": round(abs(est - d) / d * 100, 2)})
        for j, dd in enumerate(dets):
            if j in used:
                continue
            on_real = any(iou(b, dd["box"]) > 0.1 for g, b, d in ahead if g["family"] is not None)
            if not on_real:
                dd["false"] = True; false_alarms += 1
        if k in CFG.get("debug_frames", []):
            cv2.imwrite(str(OUT / f"debug_{cname}_{k}.jpg"), draw(img.copy(), cname, dets, vis, k, 0))
        if record or CFG.get("show"):
            ann = draw(img.copy(), cname, dets, vis, k, 1 / max(np.mean(lat[-20:]), 1e-6))
            if record:
                writer.write(ann)
            if CFG.get("show"):                       # live view with the model's boxes (GUI mode)
                small = cv2.resize(ann, (DW, DH), interpolation=cv2.INTER_AREA)
                if DISPLAY is not None:               # overlay inside the Webots 3D window
                    bgra = cv2.cvtColor(small, cv2.COLOR_BGR2BGRA)
                    ref = DISPLAY.imageNew(bgra.tobytes(), Display.BGRA, DW, DH)
                    DISPLAY.imagePaste(ref, 0, 0, False); DISPLAY.imageDelete(ref)
                cv2.imshow(WIN, cv2.resize(ann, (960, 540)))
                cv2.setWindowProperty(WIN, cv2.WND_PROP_TOPMOST, 1)
                cv2.waitKey(1)
        k += 1
    o = [r for r in obs_rows if r["weather"] == cname]
    e = [r["err_pct"] for r in dist_rows if r["weather"] == cname]
    wrong = sum(1 for r in o if r["pred_class"] and not r["detected"])
    ids, ids_det = {r["sign_id"] for r in o}, {r["sign_id"] for r in o if r["detected"]}
    row = {"weather": cname, "frames": k, "gt_observations": len(o), "detected_observations": sum(r["detected"] for r in o),
           "detection_rate": sum(r["detected"] for r in o) / max(len(o), 1), "signs_seen": len(ids), "signs_detected": len(ids_det),
           "sign_recall": len(ids_det) / max(len(ids), 1), "wrong_class_observations": wrong,
           "unmatched_detections": false_alarms, "unmatched_per_100_frames": 100 * false_alarms / max(k, 1),
           "median_dist_err_pct": float(np.median(e)) if e else float("nan"), "p90_dist_err_pct": float(np.percentile(e, 90)) if e else float("nan"),
           "detector_ms_p95": 1000 * float(np.percentile(lat, 95))}
    sweep.append(row)
    print(f"[{cname}] frames {k} | detection rate {row['detection_rate']:.2f} | signs {len(ids_det)}/{len(ids)} | wrong class {wrong} | "
          f"false alarms {false_alarms} | median dist err {row['median_dist_err_pct']:.1f}% | {time.time() - t_start:.0f}s", flush=True)
    write_csv(OUT / "distance_log.csv", dist_rows); write_csv(OUT / "observations.csv", obs_rows); write_csv(OUT / "weather_sweep.csv", sweep)

if writer is not None:
    writer.release()
(OUT / "README.txt").write_text("Generated by webots_demo (Webots R2025a). Columns follow the CARLA scripts; notebook section 6.4 reads distance_log.csv and weather_sweep.csv.\n")
print("done ->", OUT, flush=True)
if CFG["quit_at_end"]:
    sup.simulationQuit(0)
