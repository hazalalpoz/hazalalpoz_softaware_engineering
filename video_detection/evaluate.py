import argparse
import json
import os
from pathlib import Path

from ultralytics import YOLO

from detect_video import VideoDetector, pick_device

HERE = Path(__file__).parent
DEFAULT_MODELS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt"]


def accuracy(model_name: str, device: str) -> dict:
    model = YOLO(model_name)
    m = model.val(data="coco128.yaml", imgsz=640, batch=16, device=device, plots=False, verbose=False)
    params = sum(p.numel() for p in model.model.parameters())
    return {
        "map50": round(float(m.box.map50), 4),
        "map50_95": round(float(m.box.map), 4),
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "val_ms_per_image": {k: round(v, 2) for k, v in m.speed.items()},
        "params_m": round(params / 1e6, 1),
        "file_mb": round(os.path.getsize(model.ckpt_path) / 1024**2, 1) if model.ckpt_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--video", help="video file for the speed test")
    parser.add_argument("--frames", type=int, default=300, help="frames to process per model")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = args.device or pick_device()
    if args.video and not Path(args.video).exists():
        raise SystemExit(f"Video not found: {args.video}")

    results = []
    for name in args.models:
        print(f"\n=== {name} on '{device}' ===")
        r = {"model": name, "device": device, **accuracy(name, device)}
        print(f"COCO128: mAP50 {r['map50']:.3f}  mAP50-95 {r['map50_95']:.3f}  "
              f"P {r['precision']:.3f}  R {r['recall']:.3f}")
        if args.video:
            det = VideoDetector(name, device)
            det.process(args.video, max_frames=10, track=False)  # warm-up
            v = det.process(args.video, max_frames=args.frames, track=True)
            r["video"] = {k: v[k] for k in ("frames", "fps_end_to_end", "fps_model_only", "ms_per_frame",
                                            "avg_objects_per_frame", "unique_tracked_objects", "mean_confidence")}
            print(f"Video: {v['fps_model_only']} FPS (model), {v['fps_end_to_end']} FPS (end-to-end), "
                  f"{v['avg_objects_per_frame']} objects/frame, unique: {v['unique_tracked_objects']}")
        results.append(r)
    table = ["| Model | Params (checkpoint), M | mAP50 | mAP50-95 | Precision | Recall | Pipeline, ms/frame | FPS (model) | FPS (end-to-end) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        v = r.get("video", {})
        stages = v.get("ms_per_frame") or r["val_ms_per_image"]
        inf = round(sum(t for k, t in stages.items() if k != "loss"), 1)
        table.append(f"| {r['model']} | {r['params_m']} | {r['map50']:.3f} | {r['map50_95']:.3f} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} | {inf} | {v.get('fps_model_only', '—')} | "
                     f"{v.get('fps_end_to_end', '—')} |")
    print("\n" + "\n".join(table))

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / f"benchmark_{device}.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (out / f"benchmark_{device}.md").write_text("\n".join(table) + "\n")
    print(f"\nSaved to results/benchmark_{device}.json and .md")


if __name__ == "__main__":
    main()
