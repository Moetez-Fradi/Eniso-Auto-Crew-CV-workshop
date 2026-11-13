"""Find likely MISSING labels: confident detections that overlap no ground-truth box (of any of our 6 classes).
Writes a contact sheet per split (numbered crops with context) + candidates.csv for human review.

usage: python experiments/mine_missing_labels.py --weights models/signs6_y11s_merged.pt --data work/signs6 --out ../review [--conf 0.5]
"""
import argparse, csv
from pathlib import Path
import cv2, numpy as np


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy; u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.5); ap.add_argument("--splits", default="train,val,test")
    a = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(a.weights); names = m.names; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in a.splits.split(","):
        imgs = sorted((Path(a.data) / "images" / split).iterdir())
        for k in range(0, len(imgs), 32):
            batch = imgs[k:k + 32]
            for p, r in zip(batch, m.predict([str(x) for x in batch], conf=a.conf, imgsz=640, verbose=False, device=0)):
                H, W = r.orig_shape
                lp = Path(a.data) / "labels" / split / (p.stem + ".txt")
                gts = []
                for ln in (lp.read_text().split("\n") if lp.exists() else []):
                    if ln.strip():
                        c, x, y, w, h = map(float, ln.split()[:5])
                        gts.append(((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H))
                for box, c, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    if all(iou(box, g) < 0.3 for g in gts):
                        x1, y1, x2, y2 = box
                        rows.append(dict(split=split, file=p.name, cls=int(c), cls_name=names[int(c)], conf=round(float(s), 3),
                                         x=round((x1 + x2) / 2 / W, 6), y=round((y1 + y2) / 2 / H, 6), w=round((x2 - x1) / W, 6), h=round((y2 - y1) / H, 6)))
    for i, r in enumerate(rows):
        r["id"] = i
    with open(out / "candidates.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["id", "split", "file", "cls", "cls_name", "conf", "x", "y", "w", "h"]); wr.writeheader(); wr.writerows(rows)
    # contact sheets: crop with 50% context, numbered
    tiles = []
    for r in rows:
        im = cv2.imread(str(Path(a.data) / "images" / r["split"] / r["file"])); H, W = im.shape[:2]
        cx, cy, w, h = r["x"] * W, r["y"] * H, r["w"] * W, r["h"] * H; s = max(w, h) * 1.0
        cr = im[int(max(0, cy - s)):int(min(H, cy + s)), int(max(0, cx - s)):int(min(W, cx + s))]
        t = cv2.resize(cr, (120, 120)) if cr.size else np.zeros((120, 120, 3), np.uint8)
        t = cv2.copyMakeBorder(t, 18, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(t, f"{r['id']} {r['cls_name'][:10]} {r['conf']:.2f}", (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(t)
    per = 60
    for j in range(0, len(tiles), per):
        chunk = tiles[j:j + per]; chunk += [np.full_like(tiles[0], 255)] * ((-len(chunk)) % 10)
        sheet = np.vstack([np.hstack(chunk[i:i + 10]) for i in range(0, len(chunk), 10)])
        cv2.imwrite(str(out / f"sheet_{j // per:02d}.png"), sheet)
    from collections import Counter
    print(len(rows), "candidates", Counter(r["split"] for r in rows), Counter(r["cls_name"] for r in rows))


if __name__ == "__main__":
    main()
