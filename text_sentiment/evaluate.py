import argparse
import csv
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import psutil
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from sentiment import MODEL_NAME, SentimentAnalyzer

HERE = Path(__file__).parent
LABELS = ["negative", "neutral", "positive"]


def load_local() -> list[dict]:
    with open(HERE / "data" / "test_samples.csv", encoding="utf-8", newline="") as f:
        return [{"text": row["text"], "label": row["label"], "lang": row["lang"]}
                for row in csv.DictReader(f)]


def load_tweet_eval(n: int, seed: int = 42) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("tweet_eval", "sentiment", split="test")
    idx2label = {0: "negative", 1: "neutral", 2: "positive"}
    rng = random.Random(seed)
    sample = rng.sample(range(len(ds)), min(n, len(ds)))
    return [{"text": ds[i]["text"], "label": idx2label[ds[i]["label"]], "lang": "en"} for i in sample]


def mb(n: float) -> float:
    return round(n / 1024**2, 1)


def evaluate(model_name: str, items, device, batch_size: int) -> dict:
    print(f"\n=== {model_name} ===")
    proc = psutil.Process()
    t0 = time.perf_counter()
    sa = SentimentAnalyzer(model_name, device)
    load_s = time.perf_counter() - t0
    print(f"Loaded on '{sa.device}' in {load_s:.1f}s")

    texts = [it["text"] for it in items]
    labels = [it["label"] for it in items]
    sa.predict(texts[:4])


    lat = []
    for t in texts[:100]:
        s = time.perf_counter()
        sa.predict(t)
        lat.append((time.perf_counter() - s) * 1000)
    lat.sort()

    s = time.perf_counter()
    preds_raw = sa.predict(texts, batch_size=batch_size)
    batch_s = time.perf_counter() - s
    preds = [p["label"] for p in preds_raw]

    correct = [p == y for p, y in zip(preds, labels)]
    accuracy = sum(correct) / len(correct)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=LABELS, zero_division=0)
    macro_f1 = float(f1.mean())
    cm = confusion_matrix(labels, preds, labels=LABELS)

    by_lang = defaultdict(list)
    for it, ok in zip(items, correct):
        by_lang[it["lang"]].append(ok)
    per_lang_acc = {lang: round(sum(v) / len(v), 3) for lang, v in sorted(by_lang.items())}

    errors = [{"text": it["text"], "true": y, "pred": p}
              for it, y, p, ok in zip(items, labels, preds, correct) if not ok]

    gpu = None
    if sa.device == "mps":
        gpu = mb(torch.mps.driver_allocated_memory())
    elif sa.device == "cuda":
        gpu = mb(torch.cuda.max_memory_allocated())

    result = {
        "model": model_name, "device": sa.device, "n_texts": len(items),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {lbl: {"precision": round(float(p), 3), "recall": round(float(r), 3),
                             "f1": round(float(f), 3), "support": int(s)}
                      for lbl, p, r, f, s in zip(LABELS, precision, recall, f1, support)},
        "confusion_matrix": {"labels": LABELS, "matrix": cm.tolist()},
        "per_language_accuracy": per_lang_acc,
        "errors": errors,
        "load_time_s": round(load_s, 2),
        "latency_ms_p50": round(statistics.median(lat), 1),
        "latency_ms_p95": round(lat[int(0.95 * (len(lat) - 1))], 1),
        "throughput_texts_per_s": round(len(texts) / batch_s, 1),
        "ram_total_mb": mb(proc.memory_info().rss),
        "gpu_memory_mb": gpu,
    }

    print(f"Accuracy {accuracy:.1%} | Macro F1 {macro_f1:.1%}")
    print("Per-language accuracy:", per_lang_acc)
    print(f"Latency p50/p95 {result['latency_ms_p50']}/{result['latency_ms_p95']} ms | "
          f"throughput {result['throughput_texts_per_s']} texts/s | GPU mem {gpu} MB")
    print(f"{len(errors)} misclassified:")
    for e in errors[:10]:
        print(f"  {e['true']} -> {e['pred']}: {e['text']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", nargs="+", default=[MODEL_NAME])
    parser.add_argument("--tweet-eval", type=int, metavar="N")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    args = parser.parse_args()

    if args.tweet_eval:
        source, items = f"tweet_eval_{args.tweet_eval}", load_tweet_eval(args.tweet_eval)
    else:
        source, items = "local", load_local()
    print(f"Dataset: {source}, {len(items)} texts")

    results = [evaluate(m, items, args.device, args.batch_size) for m in args.models]

    table = ["| Model | Accuracy | Macro F1 | Latency p50, ms | Throughput, texts/s |",
             "|---|---|---|---|---|"]
    for r in results:
        table.append(f"| {r['model']} | {r['accuracy']:.1%} | {r['macro_f1']:.1%} | "
                     f"{r['latency_ms_p50']} | {r['throughput_texts_per_s']} |")
    print("\n" + "\n".join(table))

    dev = results[0]["device"]
    out = HERE / "results" / f"metrics_{source}_{dev}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
