"""Live demo: YOLO traffic-sign detection on a CARLA camera, compared with simulator ground truth.

Shows, for every sign the model finds: class, confidence, ESTIMATED distance (pinhole model) next to the TRUE distance (CARLA).
Writes carla_results/distance_log.csv and weather_sweep.csv (one row), optionally an annotated mp4.

    python carla_yolo_demo.py --weather ClearNoon --frames 600 --show --save demo.mp4
    python carla_yolo_demo.py --weights best.pt --weather Night --frames 600       # your trained 6-class model
"""
import cv2

import carla_common as cc


def main():
    p = cc.base_parser(__doc__)
    p.add_argument("--weather", default="ClearNoon", choices=cc.WEATHERS)
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--show", action="store_true", help="open a live window (ESC to stop)")
    p.add_argument("--save", default=None, help="write an annotated mp4")
    p.add_argument("--list-signs", action="store_true", help="print the sign actors of the map and exit")
    args = p.parse_args()

    model, fams, supported = cc.load_model(args.weights, args.imgsz)
    print("model classes:", model.names, "-> evaluated sign families:", sorted(supported))
    with cc.Session(args) as sess:
        signs = cc.list_signs(sess.world, supported)
        print(f"{len(signs)} ground-truth signs of the supported families in {args.town}")
        if args.list_signs:
            for s in signs:
                print(s["id"], s["family"], round(s["pos"].x, 1), round(s["pos"].y, 1))
            return
        writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (args.width, args.height)) if args.save else None
        try:
            summary, rows = cc.run_episode(sess, model, fams, supported, args.weather, args.frames, signs, show=args.show, writer=writer)
        finally:
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()})
    cc.write_csv(f"{args.out}/distance_log.csv", rows)
    cc.write_csv(f"{args.out}/weather_sweep.csv", [{"weather": args.weather, **summary}])
    print(f"wrote {args.out}/distance_log.csv ({len(rows)} rows) and weather_sweep.csv")


if __name__ == "__main__":
    main()
