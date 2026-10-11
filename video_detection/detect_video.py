import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

HERE = Path(__file__).parent
MODEL_NAME = "yolo26n.pt"

def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class VideoDetector:

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device or pick_device()
        self.model = YOLO(model_name)

    def class_ids(self, names: list[str] | None):
        if not names:
            return None
        lookup = {v: k for k, v in self.model.names.items()}
        unknown = [n for n in names if n not in lookup]
        if unknown:
            raise SystemExit(f"Unknown classes {unknown}. Available: {sorted(lookup)}")
        return [lookup[n] for n in names]

    def process(self, source: str, conf: float = 0.35, classes=None, track: bool = True,
                save_path: Path | None = None, show: bool = False, max_frames: int | None = None) -> dict:
        source = int(source) if str(source).isdigit() else source
        kwargs = dict(source=source, stream=True, conf=conf, classes=self.class_ids(classes),
                      device=self.device, verbose=False)
        stream = self.model.track(persist=True, tracker="bytetrack.yaml", **kwargs) if track \
            else self.model.predict(**kwargs)

        writer = None
        per_frame_counts, speeds = [], defaultdict(list)
        unique_ids = defaultdict(set)
        max_simultaneous = Counter()
        confidences = []
        t_start = time.perf_counter()
        n = 0
        for r in stream:
            n += 1
            for k, v in r.speed.items():
                speeds[k].append(v)
            boxes = r.boxes
            names = [r.names[int(c)] for c in boxes.cls.tolist()]
            counts = Counter(names)
            per_frame_counts.append(sum(counts.values()))
            for c, k in counts.items():
                max_simultaneous[c] = max(max_simultaneous[c], k)
            confidences += boxes.conf.tolist()
            if track and boxes.id is not None:
                for name, tid in zip(names, boxes.id.int().tolist()):
                    unique_ids[name].add(tid)

            frame = r.plot()
            if save_path is not None:
                if writer is None:
                    h, w = frame.shape[:2]
                    fps_in = 30
                    if isinstance(source, str):
                        cap = cv2.VideoCapture(source)
                        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
                        cap.release()
                    writer = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
                writer.write(frame)
            if show:
                cv2.imshow("YOLO26 (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if max_frames and n >= max_frames:
                break

        wall = time.perf_counter() - t_start
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        avg = {k: round(sum(v) / len(v), 2) for k, v in speeds.items()}
        return {
            "model": self.model_name, "device": self.device, "source": str(source),
            "frames": n,
            "fps_end_to_end": round(n / wall, 1) if wall else None,
            "ms_per_frame": avg,
            "fps_model_only": round(1000 / sum(avg.values()), 1) if avg else None,
            "avg_objects_per_frame": round(sum(per_frame_counts) / max(n, 1), 2),
            "max_objects_in_one_frame": dict(max_simultaneous.most_common()),
            "unique_tracked_objects": {k: len(v) for k, v in sorted(unique_ids.items(), key=lambda x: -len(x[1]))},
            "mean_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
            "output_video": str(save_path) if save_path else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Object detection and tracking in video")
    parser.add_argument("source", help="video file path, or camera index (0)")
    parser.add_argument("-m", "--model", default=MODEL_NAME)
    parser.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    parser.add_argument("--classes", nargs="+", help="only these classes, e.g. person car")
    parser.add_argument("--no-track", action="store_true")
    parser.add_argument("--show", action="store_true", help="show live window")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device")
    args = parser.parse_args()

    if not args.source.isdigit() and not Path(args.source).exists():
        raise SystemExit(f"Video not found: {args.source}  (files here: "
                         f"{[p.name for p in Path('.').glob('*') if p.suffix.lower() in ('.mp4', '.mov', '.avi', '.mkv')]})")

    det = VideoDetector(args.model, args.device)
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stem = "webcam" if args.source.isdigit() else Path(args.source).stem
    tag = f"{stem}_{Path(args.model).stem}"
    save = None if args.source.isdigit() else out_dir / f"annotated_{tag}.mp4"

    print(f"{args.model} on '{det.device}', processing {args.source} ...")
    stats = det.process(args.source, args.conf, args.classes, not args.no_track, save, args.show, args.max_frames)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    (out_dir / f"stats_{tag}.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nSaved stats to results/stats_{tag}.json" + (f" and video to {save}" if save else ""))


if __name__ == "__main__":
    main()
