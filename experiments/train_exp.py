"""Train one candidate on the merged dataset and compare it per test set and per class.

usage: python experiments/train_exp.py --model yolov8n.pt --name v8n_merged [--epochs 50] [--imgsz 640]
       python experiments/train_exp.py --eval-only path/to/best.pt --name old_model
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work_exp" / "merged"


def evaluate(weights, name, imgsz):
    from ultralytics import YOLO
    m = YOLO(str(weights)); res = {}
    for test in ["test_a", "test_b"]:
        r = m.val(data=str(DATA / f"{test}.yaml"), split="test", imgsz=imgsz, batch=16, workers=4, plots=False, verbose=False,
                  project=str(ROOT / "work_exp" / "val"), name=f"{name}_{test}", exist_ok=True)
        names = r.names
        per = {names[c]: dict(p=round(float(r.box.p[i]), 3), r=round(float(r.box.r[i]), 3), ap50=round(float(r.box.ap50[i]), 3))
               for i, c in enumerate(r.box.ap_class_index)}
        res[test] = dict(map50=round(float(r.box.map50), 3), map=round(float(r.box.map), 3), per_class=per)
        print(f"\n== {name} on {test}: mAP50 {res[test]['map50']}  mAP50-95 {res[test]['map']}")
        for k, v in per.items():
            print(f"   {k:15s} P {v['p']:.3f}  R {v['r']:.3f}  AP50 {v['ap50']:.3f}")
    out = ROOT / "work_exp" / "results"; out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / f"{name}.json", "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt"); ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--eval-only", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data-yaml", default=None, help="train on this yaml (e.g. work/signs6/data.yaml from build_data_only.py); skips the test_a/test_b evaluation")
    ap.add_argument("--resume", action="store_true", help="continue runs/<name>/weights/last.pt (e.g. after an out-of-memory crash)")
    a = ap.parse_args()
    if a.eval_only:
        evaluate(a.eval_only, a.name, a.imgsz); return
    from ultralytics import YOLO
    if a.resume:
        YOLO(str(ROOT / "work_exp" / "runs" / a.name / "weights" / "last.pt")).train(resume=True, workers=a.workers)
        evaluate(ROOT / "work_exp" / "runs" / a.name / "weights" / "best.pt", a.name, a.imgsz); return
    m = YOLO(a.model)
    m.train(data=str(a.data_yaml or DATA / "merged.yaml"), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, patience=20, seed=42,
            fliplr=0.0, degrees=5, scale=0.5, close_mosaic=10, workers=a.workers, device=0, cache=False,
            project=str(ROOT / "work_exp" / "runs"), name=a.name, exist_ok=True, plots=True)
    best = ROOT / "work_exp" / "runs" / a.name / "weights" / "best.pt"
    if not best.exists():   # ultralytics may nest relative projects under runs/detect
        best = next((ROOT / "work_exp").rglob(f"{a.name}/weights/best.pt"))
    if a.data_yaml:
        print("best weights:", best); return
    evaluate(best, a.name, a.imgsz)


if __name__ == "__main__":
    main()
