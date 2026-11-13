"""Closed-loop robustness test: the SAME drive repeated under different weather / lighting, with ground-truth scoring.

This is the simulator version of the slice evaluation from Step 3: detection rate of visible signs per condition.

    python carla_weather_sweep.py --frames 400
    python carla_weather_sweep.py --weights best.pt --weathers ClearNoon Night HardRainNoon FogNoon

Outputs carla_results/weather_sweep.csv and distance_log.csv, which the workshop notebook (section 6.4) reads.
"""
import carla_common as cc


def main():
    p = cc.base_parser(__doc__)
    p.add_argument("--frames", type=int, default=400, help="frames per weather")
    p.add_argument("--weathers", nargs="+", default=cc.WEATHERS, choices=cc.WEATHERS)
    args = p.parse_args()

    model, fams, supported = cc.load_model(args.weights, args.imgsz)
    print("model classes:", model.names, "-> evaluated sign families:", sorted(supported))
    sweep, all_rows = [], []
    with cc.Session(args) as sess:
        signs = cc.list_signs(sess.world, supported)
        print(f"{len(signs)} ground-truth signs in {args.town}")
        for w in args.weathers:
            print(f"--- {w} ...", flush=True)
            summary, rows = cc.run_episode(sess, model, fams, supported, w, args.frames, signs)
            sweep.append({"weather": w, **summary}); all_rows += rows
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()})
    cc.write_csv(f"{args.out}/weather_sweep.csv", sweep)
    cc.write_csv(f"{args.out}/distance_log.csv", all_rows)
    print(f"wrote {args.out}/weather_sweep.csv and distance_log.csv")


if __name__ == "__main__":
    main()
