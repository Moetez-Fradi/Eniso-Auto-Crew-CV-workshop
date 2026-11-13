# From Prototype to Production: Computer Vision for Autonomous Cars

A notebook that takes a traffic-sign detector (Stop, Speed Limit 30/50, No Entry, Give Way, Keep Right) through the six steps
a real team goes through before a model goes into a car: **requirements → data → validation → deployment → perception stack → monitoring**.

## Start

* **Run it:** [open the notebook in Colab](https://colab.research.google.com/github/Moetez-Fradi/Eniso-Auto-Crew-CV-workshop/blob/main/cv_production_workshop.ipynb),
  choose *Runtime → Change runtime type → T4 GPU*, add your Kaggle API token as a Colab secret named `KAGGLE_API_TOKEN`, and run the cells top to bottom.
  The trained models come with the repo, so nothing is trained.
* **Just read it:** `cv_production_workshop_executed.ipynb` has every output from a full run.

## What's inside

| Path | What it is |
|---|---|
| `cv_production_workshop.ipynb` | the notebook |
| `cv_production_workshop_executed.ipynb` | the same, with all outputs |
| `models/` | trained weights: the candidate (YOLO11s) and the original prototype (YOLOv8n) |
| `label_fixes.csv` | 136 reviewed label corrections applied by the notebook |
| `webots_demo/` | simulation: a car drives past the signs in 6 lighting conditions, with true distances ([README](webots_demo/README.md)) |
| `notebook_src/`, `build_notebook.py` | the notebook as plain Python, and the script that rebuilds it |
| `experiments/` | scripts used to choose and train the model outside the notebook |
| `carla_demo/` | the same simulation idea for CARLA (not tested) |

Running locally instead of Colab: install PyTorch for your GPU, then `pip install -r requirements.txt`.

## Result

Release gate: **SHIP**. Stop recall 1.00, other classes ≥ 0.91, 3.7 false alarms per 100 sign-free images, 27 ms p95 latency (RTX 5050 laptop GPU), 6.8 % distance error in simulation.
The requirements are illustrative, chosen for learning; the notebook explains each one and where the evidence is weak.

## Credits

Prototype: *SignSight ADAS* by Malak Chartaoui (ENISo). Data (downloaded by the notebook, not included): [Traffic Sign Localization & Detection](https://www.kaggle.com/datasets/ahemateja19bec1025/trafficsignlocalizationdetectionyoloannotated) (Kaggle) and
*Street Sign Set* by Alessandro Ferrante (CC BY 4.0). Built with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [Webots](https://cyberbotics.com).
