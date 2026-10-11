import argparse
import json
import random
import shutil
import statistics
import subprocess
import tarfile
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import psutil
import torch
from PIL import Image

from classify import MODEL_NAME, ImageClassifier

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"

# Imagenette folder (WordNet ID) -> ImageNet-1K class index
WNID_TO_IDX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}


def class_counts(root: Path) -> dict[str, int]:
    return {w: len(list((root / "val" / w).glob("*.JPEG"))) for w in WNID_TO_IDX}


def download() -> Path:
    root = DATA_DIR / "imagenette2-320"
    if (root / "val").exists() and all(class_counts(root).values()):
        return root
    DATA_DIR.mkdir(exist_ok=True)
    tgz = DATA_DIR / "imagenette2-320.tgz"
    if not tgz.exists():
        print(f"Downloading Imagenette (~330 MB) from {URL}")
        urllib.request.urlretrieve(URL, tgz, lambda b, bs, t: print(f"\r  {b * bs / t:5.1%}", end=""))
        print()
    print("Extracting validation split ...")
    if shutil.which("tar"):
        subprocess.run(["tar", "-xzf", str(tgz), "-C", str(DATA_DIR), "imagenette2-320/val"], check=True)
    else:
        with tarfile.open(tgz) as tar:
            members = [m for m in tar.getmembers() if "/val/" in m.name or m.name.endswith("/val")]
            tar.extractall(DATA_DIR, members=members, filter="data")
    counts = class_counts(root)
    print("Images per class:", counts)
    missing = [w for w, c in counts.items() if c == 0]
    if missing:
        raise SystemExit(f"No images for {missing} after extraction — the archive is probably incomplete. "
                         f"Delete the folder {DATA_DIR} and run again to re-download.")
    return root


def load_items(per_class: int, seed: int = 42) -> list[tuple[Path, int]]:
    root = download() / "val"
    rng = random.Random(seed)
    items = []
    for wnid, idx in WNID_TO_IDX.items():
        files = sorted((root / wnid).glob("*.JPEG"))
        if len(files) < per_class:
            print(f"  warning: only {len(files)} images for {wnid}")
        items += [(f, idx) for f in rng.sample(files, min(per_class, len(files)))]
    return items


def mb(n: float) -> float:
    return round(n / 1024**2, 1)


def evaluate(model_name: str, items, device, batch_size: int) -> dict:
    print(f"\n=== {model_name} ===")
    proc = psutil.Process()
    t0 = time.perf_counter()
    clf = ImageClassifier(model_name, device)
    load_s = time.perf_counter() - t0
    info = clf.info
    print(f"Loaded on '{clf.device}' in {load_s:.1f}s | {info['params_m']}M params, "
          f"{info['gflops']} GFLOPs | official ImageNet top-1 {info['imagenet_top1']}%")

    images = [Image.open(p).convert("RGB") for p, _ in items]
    labels = torch.tensor([y for _, y in items])
    clf.logits(images[:4])

    lat = []
    for img in images[:100]:
        s = time.perf_counter()
        clf.logits([img])
        lat.append((time.perf_counter() - s) * 1000)
    lat.sort()

    s = time.perf_counter()
    logits = torch.cat([clf.logits(images[i:i + batch_size]) for i in range(0, len(images), batch_size)])
    batch_s = time.perf_counter() - s

    top5 = logits.topk(5, dim=1).indices
    top1 = top5[:, 0]
    top1_acc = (top1 == labels).float().mean().item()
    top5_acc = (top5 == labels[:, None]).any(dim=1).float().mean().item()


    idx10 = torch.tensor(list(WNID_TO_IDX.values()))
    pred10 = idx10[logits[:, idx10].argmax(dim=1)]
    acc10 = (pred10 == labels).float().mean().item()

    per_class = defaultdict(lambda: [0, 0])
    confusions = Counter()
    for y, p in zip(labels.tolist(), top1.tolist()):
        per_class[y][0] += y == p
        per_class[y][1] += 1
        if y != p:
            confusions[(clf.categories[y], clf.categories[p])] += 1

    gpu = None
    if clf.device == "mps":
        gpu = mb(torch.mps.driver_allocated_memory())
    elif clf.device == "cuda":
        gpu = mb(torch.cuda.max_memory_allocated())

    result = {
        "model": model_name, "device": clf.device, "n_images": len(items), **info,
        "top1_acc_1000": round(top1_acc, 4),
        "top5_acc_1000": round(top5_acc, 4),
        "acc_10_classes": round(acc10, 4),
        "per_class_top1": {clf.categories[k]: round(c / n, 3) for k, (c, n) in per_class.items()},
        "top_confusions": [{"true": t, "pred": p, "count": c} for (t, p), c in confusions.most_common(10)],
        "load_time_s": round(load_s, 2),
        "latency_ms_p50": round(statistics.median(lat), 1),
        "latency_ms_p95": round(lat[int(0.95 * (len(lat) - 1))], 1),
        "throughput_img_per_s": round(len(images) / batch_s, 1),
        "ram_total_mb": mb(proc.memory_info().rss),
        "gpu_memory_mb": gpu,
    }

    print(f"Top-1 {top1_acc:.1%} | Top-5 {top5_acc:.1%} | 10-class accuracy {acc10:.1%}")
    print("Per-class top-1:", {k: f"{v:.0%}" for k, v in result["per_class_top1"].items()})
    print("Most frequent confusions (true -> predicted):")
    for c in result["top_confusions"][:6]:
        print(f"  {c['count']:3}x  {c['true']} -> {c['pred']}")
    print(f"Latency p50/p95 {result['latency_ms_p50']}/{result['latency_ms_p95']} ms | "
          f"throughput {result['throughput_img_per_s']} img/s | GPU mem {gpu} MB")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", nargs="+", default=[MODEL_NAME])
    parser.add_argument("--per-class", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    args = parser.parse_args()

    items = load_items(args.per_class)
    print(f"Dataset: Imagenette val, {len(items)} images ({args.per_class} per class)")
    results = [evaluate(m, items, args.device, args.batch_size) for m in args.models]

    table = ["| Model | Params, M | GFLOPs | Top-1 (1000 cl.) | Top-5 | 10-class acc. | Latency p50, ms | Throughput, img/s |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        table.append(f"| {r['model']} | {r['params_m']} | {r['gflops']} | {r['top1_acc_1000']:.1%} | "
                     f"{r['top5_acc_1000']:.1%} | {r['acc_10_classes']:.1%} | {r['latency_ms_p50']} | "
                     f"{r['throughput_img_per_s']} |")
    print("\n" + "\n".join(table))

    dev = results[0]["device"]
    out = HERE / "results" / f"metrics_imagenette_{dev}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
