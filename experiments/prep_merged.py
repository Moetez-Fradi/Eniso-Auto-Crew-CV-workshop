"""Build a merged 6-class YOLO dataset from
   A = original SignSight source (Kaggle ahemateja19bec1025/trafficsignlocalizationdetectionyoloannotated), folder data/
   B = Street Sign Set (Ferrante 2025, CC BY 4.0, huggingface.co/datasets/AlessandroFerrante/StreetSignSet), folder data_sss/

Every other sign class is dropped from the labels, so those signs become hard negatives.
Test splits are also written per source (test_a, test_b) so we can compare against the old model on the exact old test set.

usage: python experiments/prep_merged.py [--out work_exp/merged]
"""
import argparse, shutil, re
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
TARGET = ["Stop", "Speed Limit 30", "Speed Limit 50", "No Entry", "Give Way", "Keep Right"]

# source class name -> target class. A's "Turn Right" is mostly the blue *keep right* sign (checked visually).
ALIASES = {
    "stop": "Stop", "prio_stop": "Stop",
    "speed limit 30": "Speed Limit 30", "forb_speed_over_30": "Speed Limit 30",
    "speed limit 50": "Speed Limit 50", "forb_speed_over_50": "Speed Limit 50",
    "no entry": "No Entry", "forb_no_entry": "No Entry",
    "give way": "Give Way", "prio_give_way": "Give Way",
    "turn right": "Keep Right", "mand_pass_right": "Keep Right",
}
SOURCES = {
    "a": dict(root=ROOT / "data", splits={"train": "train", "val": "valid", "test": "test"}),
    "b": dict(root=ROOT / "data_sss", splits={"train": "train", "val": "valid", "test": "test"}),
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def find_split_dir(root, split):
    for p in root.rglob("images"):
        if p.is_dir() and p.parent.name == split:
            return p
    raise FileNotFoundError(f"{split}/images under {root}")


def load_names(root):
    y = next(p for p in root.rglob("*.yaml"))
    names = yaml.safe_load(open(y, encoding="utf8"))["names"]
    return names if isinstance(names, list) else [names[k] for k in sorted(names)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "work_exp" / "merged")); a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    stats = {}
    for tag, src in SOURCES.items():
        names = load_names(src["root"])
        remap = {i: TARGET.index(ALIASES[norm(n)]) for i, n in enumerate(names) if norm(n) in ALIASES}
        print(f"source {tag}: {len(names)} classes, mapped:", {names[i]: TARGET[j] for i, j in remap.items()})
        for split, sdir in src["splits"].items():
            img_dir = find_split_dir(src["root"], sdir)
            dests = [split] + ([f"test_{tag}"] if split == "test" else [])
            n_img = n_box = 0
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXT:
                    continue
                lbl = img.parent.parent / "labels" / (img.stem + ".txt")
                lines = []
                if lbl.exists():
                    for ln in lbl.read_text().splitlines():
                        p = ln.split()
                        if len(p) == 5 and int(p[0]) in remap:
                            lines.append(" ".join([str(remap[int(p[0])])] + p[1:]))
                for d in dests:
                    (out / "images" / d).mkdir(parents=True, exist_ok=True); (out / "labels" / d).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(img, out / "images" / d / f"{tag}_{img.name}")
                    (out / "labels" / d / f"{tag}_{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                n_img += 1; n_box += len(lines)
            stats[(tag, split)] = (n_img, n_box)
    for k, (ni, nb) in stats.items():
        print(f"{k[0]} {k[1]:5s}: {ni:5d} images, {nb:5d} target boxes")
    for name, test in [("merged", "test"), ("test_a", "test_a"), ("test_b", "test_b")]:
        yaml.safe_dump({"path": str(out), "train": "images/train", "val": "images/val" if name == "merged" else f"images/{test}",
                        "test": f"images/{test}", "names": dict(enumerate(TARGET))}, open(out / f"{name}.yaml", "w"))
    print("wrote", out)


if __name__ == "__main__":
    main()
