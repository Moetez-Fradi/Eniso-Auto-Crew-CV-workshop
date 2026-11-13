"""Shared helpers for the CARLA demos (YOLO traffic-sign detection with ground-truth evaluation).

The geometry / matching / summary functions at the top do not need CARLA and are unit-tested separately.
Everything that touches the simulator is below the "CARLA" marker.
"""
import argparse
import csv
import math
import queue
import random
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import carla
except ImportError:                      # lets `--help` and the pure functions work without CARLA installed
    carla = None

# ----------------------------------------------------------------------------- sign families
# CARLA blueprint id -> family used to compare simulator ground truth with detector classes
CARLA_FAMILY = {
    "traffic.stop": "stop",
    "traffic.yield": "yield",
    "traffic.speed_limit.30": "speed_30",
    "traffic.speed_limit.50": "speed_50",
}


def det_family(class_name):
    """Map a detector class name ('stop sign', 'Stop', 'Speed Limit 30', 'Give Way', ...) to a family, or None."""
    n = class_name.lower().replace("_", " ")
    if "stop" in n and "stopping" not in n:
        return "stop"
    if "give way" in n or "yield" in n:
        return "yield"
    if "speed" in n and "30" in n:
        return "speed_30"
    if "speed" in n and "50" in n:
        return "speed_50"
    return None


# ----------------------------------------------------------------------------- pure geometry / evaluation
def pinhole_distance(f_px, real_h_m, box_h_px):
    return f_px * real_h_m / max(float(box_h_px), 1e-6)


def inflate(box, fx=0.5, fy=1.0):
    """Grow an (x1,y1,x2,y2) box by a fraction of its size on each side (the simulator's sign anchor is not the sign face centre)."""
    w, h = box[2] - box[0], box[3] - box[1]
    return (box[0] - fx * w, box[1] - fy * h, box[2] + fx * w, box[3] + fy * h)


def associate(visible_gt, dets):
    """Match detections to visible ground-truth signs.

    visible_gt: [{'id', 'family', 'u', 'v', 'depth'}]    (projected pixel position and depth of each sign)
    dets:       [{'box', 'family', 'conf'}]
    Returns (matches, unmatched_det_indices) with matches = {gt_index: det_index}; one detection serves one sign.
    """
    cand = []
    for gi, g in enumerate(visible_gt):
        for di, d in enumerate(dets):
            if d["family"] != g["family"]:
                continue
            x1, y1, x2, y2 = inflate(d["box"])
            if x1 <= g["u"] <= x2 and y1 <= g["v"] <= y2:
                cx, cy = (d["box"][0] + d["box"][2]) / 2, (d["box"][1] + d["box"][3]) / 2
                cand.append((math.hypot(cx - g["u"], cy - g["v"]) - 5.0 * d["conf"], gi, di))
    matches, used_d = {}, set()
    for _, gi, di in sorted(cand):
        if gi in matches or di in used_d:
            continue
        matches[gi] = di; used_d.add(di)
    return matches, [i for i in range(len(dets)) if i not in used_d]


def summarize(rows, frames, unmatched):
    """rows: per sign observation dicts with 'detected' (bool), 'sign_id', 'err_pct' (or None)."""
    n = len(rows)
    det = sum(r["detected"] for r in rows)
    ids = {r["sign_id"] for r in rows}
    ids_det = {r["sign_id"] for r in rows if r["detected"]}
    errs = [r["err_pct"] for r in rows if r["err_pct"] is not None]
    return {
        "frames": frames, "gt_observations": n, "detected_observations": det,
        "detection_rate": det / n if n else float("nan"),
        "signs_seen": len(ids), "signs_detected": len(ids_det), "sign_recall": len(ids_det) / len(ids) if ids else float("nan"),
        "unmatched_detections": unmatched, "unmatched_per_100_frames": 100 * unmatched / max(frames, 1),
        "median_dist_err_pct": float(np.median(errs)) if errs else float("nan"),
        "p90_dist_err_pct": float(np.percentile(errs, 90)) if errs else float("nan"),
    }


# ----------------------------------------------------------------------------- CLI
def base_parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", default="Town03", help="map to load (Town03 has many signs)")
    p.add_argument("--weights", default="yolov8n.pt", help="COCO yolov8n.pt (detects 'stop sign' only) or your trained best.pt (6 classes)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--sign-h", type=float, default=0.75, help="assumed real sign height in metres for the pinhole distance")
    p.add_argument("--spawn", type=int, default=-1, help="spawn point index; -1 = automatically pick the one with most signs ahead")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fov", type=float, default=90.0)
    p.add_argument("--min-depth", type=float, default=5.0)
    p.add_argument("--max-depth", type=float, default=40.0)
    p.add_argument("--sign-front", choices=["forward", "backward", "any"], default="forward",
                   help="which side of a CARLA sign actor is its readable face (check once with --show: green dots must sit on signs facing you)")
    p.add_argument("--out", default="carla_results")
    return p


# ----------------------------------------------------------------------------- CARLA
WEATHERS = ["ClearNoon", "CloudyNoon", "WetNoon", "HardRainNoon", "ClearSunset", "Night", "FogNoon"]


def make_weather(name):
    if name == "Night":
        return carla.WeatherParameters(cloudiness=10.0, precipitation=0.0, precipitation_deposits=0.0, wind_intensity=5.0,
                                       sun_azimuth_angle=0.0, sun_altitude_angle=-30.0, fog_density=0.0)
    if name == "FogNoon":
        return carla.WeatherParameters(cloudiness=20.0, precipitation=0.0, precipitation_deposits=0.0, wind_intensity=5.0,
                                       sun_azimuth_angle=0.0, sun_altitude_angle=60.0, fog_density=60.0)
    return getattr(carla.WeatherParameters, name)


class Session:
    """Connects, switches to synchronous mode, and restores the simulator on exit."""

    def __init__(self, args):
        if carla is None:
            raise SystemExit("The 'carla' Python package is not installed (see README.md).")
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(60.0)
        self.world = self.client.load_world(args.town)
        self.orig = self.world.get_settings()
        s = self.world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = 0.05
        self.world.apply_settings(s)
        self.tm = self.client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        self.tm.set_random_device_seed(args.seed)
        random.seed(args.seed)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.tm.set_synchronous_mode(False)
            self.world.apply_settings(self.orig)
        except Exception:
            pass


def list_signs(world, supported):
    out = []
    for a in world.get_actors().filter("traffic.*"):
        fam = CARLA_FAMILY.get(a.type_id)
        if fam and fam in supported:
            tf = a.get_transform()
            try:
                pos = tf.transform(a.bounding_box.location)
            except Exception:
                pos = tf.location
            out.append({"id": a.id, "family": fam, "pos": pos, "forward": tf.get_forward_vector()})
    return out


def choose_spawn(world, signs, min_ahead=8.0, max_ahead=60.0, max_side=10.0):
    best, best_n = 0, -1
    pts = world.get_map().get_spawn_points()
    for i, sp in enumerate(pts):
        f, loc = sp.get_forward_vector(), sp.location
        n = 0
        for s in signs:
            dx, dy = s["pos"].x - loc.x, s["pos"].y - loc.y
            ahead = dx * f.x + dy * f.y
            side = abs(-dx * f.y + dy * f.x)
            n += (min_ahead < ahead < max_ahead) and side < max_side
        if n > best_n:
            best, best_n = i, n
    return best, pts[best]


class CameraRig:
    def __init__(self, world, vehicle, args, x=1.5, z=1.6):
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        for k, v in (("image_size_x", args.width), ("image_size_y", args.height), ("fov", args.fov)):
            bp.set_attribute(k, str(v))
        self.actor = world.spawn_actor(bp, carla.Transform(carla.Location(x=x, z=z)), attach_to=vehicle)
        self.q = queue.Queue()
        self.actor.listen(self.q.put)
        self.w, self.h = args.width, args.height
        self.f = self.w / (2.0 * math.tan(math.radians(args.fov) / 2.0))

    def grab(self, frame_id, timeout=10.0):
        while True:
            img = self.q.get(timeout=timeout)
            if img.frame >= frame_id:
                arr = np.frombuffer(img.raw_data, np.uint8).reshape(self.h, self.w, 4)[:, :, :3]
                return arr.copy()

    def project(self, pos, cam_tf):
        """World point -> (u, v, depth) in pixels / metres, or None if behind the camera."""
        inv = np.array(cam_tf.get_inverse_matrix())
        p = inv @ np.array([pos.x, pos.y, pos.z, 1.0])
        xf, yr, zu = p[0], p[1], p[2]            # CARLA/UE: x forward, y right, z up
        if xf <= 0.1:
            return None
        return self.f * yr / xf + self.w / 2, self.f * (-zu) / xf + self.h / 2, xf

    def destroy(self):
        try:
            self.actor.stop()
        finally:
            self.actor.destroy()


def load_model(weights, imgsz):
    from ultralytics import YOLO
    m = YOLO(weights)
    m.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz, verbose=False)
    fams = {i: det_family(n) for i, n in m.names.items()}
    supported = {f for f in fams.values() if f}
    if not supported:
        raise SystemExit(f"None of the classes of {weights} map to a sign family: {m.names}")
    return m, fams, supported


def detect(model, fams, frame, imgsz, conf):
    r = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
    out = []
    for b, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
        if fams.get(int(c)):
            out.append({"box": tuple(float(v) for v in b), "family": fams[int(c)], "name": model.names[int(c)], "conf": float(s)})
    return out


def facing_ok(sign_fwd, cam_fwd, mode):
    if mode == "any":
        return True
    dot = sign_fwd.x * cam_fwd.x + sign_fwd.y * cam_fwd.y + sign_fwd.z * cam_fwd.z
    return dot < -0.3 if mode == "forward" else dot > 0.3


def run_episode(sess, model, fams, supported, weather, n_frames, signs, show=False, writer=None, warmup=15):
    """Drives n_frames in the given weather. Returns (summary dict, distance rows)."""
    args, world = sess.args, sess.world
    world.set_weather(make_weather(weather))
    idx, spawn = (args.spawn, world.get_map().get_spawn_points()[args.spawn]) if args.spawn >= 0 else choose_spawn(world, signs)
    bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    vehicle = world.try_spawn_actor(bp, spawn)
    if vehicle is None:
        raise RuntimeError(f"could not spawn the vehicle at spawn point {idx}; try another --spawn")
    rig = None
    obs, dist_rows, unmatched, ema = [], [], 0, {}
    try:
        vehicle.set_autopilot(True, sess.tm.get_port())
        sess.tm.vehicle_percentage_speed_difference(vehicle, 40.0)              # drive slower: more frames per sign
        if weather in ("Night", "FogNoon"):
            try:
                vehicle.set_light_state(carla.VehicleLightState(carla.VehicleLightState.LowBeam))
            except Exception:
                pass
        rig = CameraRig(world, vehicle, args)
        t_fps = time.perf_counter(); fps = 0.0
        for k in range(n_frames + warmup):
            fid = world.tick()
            frame = rig.grab(fid)
            if k < warmup:                                                       # camera exposure / weather need a few ticks
                continue
            t0 = time.perf_counter()
            dets = detect(model, fams, frame, args.imgsz, args.conf)
            fps = 0.9 * fps + 0.1 / max(time.perf_counter() - t0, 1e-6) if fps else 1 / max(time.perf_counter() - t0, 1e-6)
            cam_tf = rig.actor.get_transform()
            cam_fwd = cam_tf.get_forward_vector()
            visible = []
            for s in signs:
                pr = rig.project(s["pos"], cam_tf)
                if pr is None:
                    continue
                u, v, d = pr
                if not (args.min_depth <= d <= args.max_depth and 20 < u < rig.w - 20 and 20 < v < rig.h - 20):
                    continue
                if not facing_ok(s["forward"], cam_fwd, args.sign_front):
                    continue
                visible.append({"id": s["id"], "family": s["family"], "u": u, "v": v, "depth": d})
            matches, unm = associate(visible, dets)
            unmatched += len(unm)
            for gi, g in enumerate(visible):
                row = {"sign_id": g["id"], "family": g["family"], "detected": gi in matches, "err_pct": None}
                if gi in matches:
                    det = dets[matches[gi]]
                    est = pinhole_distance(rig.f, args.sign_h, det["box"][3] - det["box"][1])
                    ema[g["id"]] = est if g["id"] not in ema else 0.7 * ema[g["id"]] + 0.3 * est
                    row["err_pct"] = abs(est - g["depth"]) / g["depth"] * 100
                    dist_rows.append({"frame": k - warmup, "weather": weather, "sign_id": g["id"], "family": g["family"], "conf": round(det["conf"], 3),
                                      "gt_dist_m": round(g["depth"], 2), "est_dist_m": round(est, 2), "est_dist_smoothed_m": round(ema[g["id"]], 2),
                                      "box_h_px": round(det["box"][3] - det["box"][1], 1)})
                obs.append(row)
            if show or writer is not None:
                vis = frame.copy()
                for g in visible:
                    cv2.circle(vis, (int(g["u"]), int(g["v"])), 6, (0, 255, 0), -1)
                    cv2.putText(vis, f"GT {g['depth']:.1f}m", (int(g["u"]) + 8, int(g["v"])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                for gi, di in matches.items():
                    d, g = dets[di], visible[gi]
                    x1, y1, x2, y2 = [int(v) for v in d["box"]]
                    est = pinhole_distance(rig.f, args.sign_h, y2 - y1)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(vis, f"{d['name']} {d['conf']:.2f} est {est:.1f}m", (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                for di in unm:
                    x1, y1, x2, y2 = [int(v) for v in dets[di]["box"]]
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 1)
                cv2.putText(vis, f"{weather} | FPS {fps:.1f} | green=ground truth, red=detection, orange=unmatched", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if writer is not None:
                    writer.write(vis)
                if show:
                    cv2.imshow("CARLA + YOLO", vis)
                    if cv2.waitKey(1) & 0xFF == 27:                             # ESC
                        break
    finally:
        if rig is not None:
            rig.destroy()
        vehicle.destroy()
        world.tick()
    return summarize(obs, n_frames, unmatched), dist_rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
