import argparse
import csv
import json
import os
import random
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import psutil
import requests
import torch

from transcribe import MODEL_NAME, SpeechRecognizer, load_audio

HERE = Path(__file__).parent
ROWS_API = "https://datasets-server.huggingface.co/rows"


def _get_page(offset: int, tries: int = 4) -> dict | None:
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    params = {"dataset": "openslr/librispeech_asr", "config": "clean", "split": "test",
              "offset": offset, "length": 100}
    for attempt in range(tries):
        try:
            r = requests.get(ROWS_API, params=params, headers=headers, timeout=60)
            if r.ok:
                return r.json()
            print(f"  page {offset}: HTTP {r.status_code}, retrying ...")
        except requests.RequestException as e:
            print(f"  page {offset}: {e.__class__.__name__}, retrying ...")
        time.sleep(3 * (attempt + 1))
    return None


def load_librispeech(n: int, seed: int = 42) -> list[dict]:
    cache = HERE / "data" / "librispeech"
    cache.mkdir(parents=True, exist_ok=True)
    meta_file = cache / "meta.json"
    if meta_file.exists():
        rows = json.loads(meta_file.read_text())
    else:
        print("Fetching LibriSpeech test-clean index ...")
        first = _get_page(0)
        if first is None:
            raise SystemExit("Hugging Face datasets-server is not responding. Try again later.")
        total = first["num_rows_total"]
        rows, failed = [], []
        for offset in range(0, total, 100):
            data = first if offset == 0 else _get_page(offset)
            if data is None:
                failed.append(offset)
                continue
            rows += [{"id": x["row"]["id"], "text": x["row"]["text"], "url": x["row"]["audio"][0]["src"]}
                     for x in data["rows"]]
        if failed:
            print(f"  skipped pages {failed} (server errors); using {len(rows)}/{total} rows")
        else:
            meta_file.write_text(json.dumps(rows))

    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    items = []
    for i, row in enumerate(sample):
        path = cache / f"{row['id']}.flac"
        if not path.exists():
            print(f"\rDownloading audio {i + 1}/{len(sample)}", end="", flush=True)
            try:
                audio = requests.get(row["url"], timeout=60)
                audio.raise_for_status()
            except requests.RequestException:
                print(f"\n  skipped {row['id']} (download failed)")
                continue
            path.write_bytes(audio.content)
        items.append({"file": path, "text": row["text"], "lang": "en", "language": "english"})
    print()
    return items


LANG_NAMES = {"en": "english", "ru": "russian", "tr": "turkish"}


def load_own() -> list[dict]:
    items = []
    with open(HERE / "data" / "recordings.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            path = HERE / "data" / "recordings" / row["file"]
            matches = list(path.parent.glob(path.stem + ".*"))  # allow .m4a/.wav/.mp3
            matches = [m for m in matches if m.suffix != ".gitkeep"]
            if not matches:
                print(f"  skipped (not recorded yet): {row['file']}")
                continue
            items.append({"file": matches[0], "text": row["text"], "lang": row["lang"],
                          "language": LANG_NAMES.get(row["lang"])})
    if not items:
        raise SystemExit("No recordings found in data/recordings/. See data/recordings.csv.")
    return items


def basic_normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")
    text = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def make_normalizer(asr: SpeechRecognizer):
    tok = asr.pipe.tokenizer

    def norm(text: str, lang: str) -> str:
        if lang == "en":
            try:
                return basic_normalize(tok.normalize(text))
            except Exception:
                pass
        return basic_normalize(re.sub(r"['’ʼ]", "", text))
    return norm


def edit_distance(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1,
                         cur[j - 1] + 1,
                         prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def mb(n: float) -> float:
    return round(n / 1024**2, 1)

def evaluate(model: str, items: list[dict], device: str | None) -> dict:
    print(f"\n=== {model} ===")
    proc = psutil.Process()
    t0 = time.perf_counter()
    asr = SpeechRecognizer(model, device)
    load_s = time.perf_counter() - t0
    print(f"Loaded on '{asr.device}' in {load_s:.1f}s")
    norm = make_normalizer(asr)

    asr.transcribe(load_audio(items[0]["file"]), items[0]["language"])

    per_lang = defaultdict(lambda: {"w_err": 0, "w_ref": 0, "c_err": 0, "c_ref": 0, "audio_s": 0.0, "proc_s": 0.0})
    samples = []
    for i, it in enumerate(items, 1):
        audio = load_audio(it["file"])
        r = asr.transcribe(audio, it["language"])
        ref, hyp = norm(it["text"], it["lang"]), norm(r["text"], it["lang"])
        w_err = edit_distance(ref.split(), hyp.split())
        c_err = edit_distance(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
        s = per_lang[it["lang"]]
        s["w_err"] += w_err; s["w_ref"] += len(ref.split())
        s["c_err"] += c_err; s["c_ref"] += len(ref.replace(" ", ""))
        s["audio_s"] += r["audio_s"]; s["proc_s"] += r["processing_s"]
        samples.append({"file": it["file"].name, "lang": it["lang"], "reference": it["text"],
                        "hypothesis": r["text"], "wer": round(w_err / max(len(ref.split()), 1), 3),
                        "audio_s": r["audio_s"], "processing_s": r["processing_s"]})
        print(f"\r  {i}/{len(items)} transcribed", end="", flush=True)
    print()

    def summarize(s):
        return {"wer": round(s["w_err"] / max(s["w_ref"], 1), 4),
                "cer": round(s["c_err"] / max(s["c_ref"], 1), 4),
                "audio_s": round(s["audio_s"], 1), "rtf": round(s["proc_s"] / s["audio_s"], 3)}

    total = defaultdict(float)
    for s in per_lang.values():
        for k, v in s.items():
            total[k] += v

    gpu = None
    if asr.device == "mps":
        gpu = mb(torch.mps.driver_allocated_memory())
    elif asr.device == "cuda":
        gpu = mb(torch.cuda.max_memory_allocated())

    overall = summarize(total)
    result = {
        "model": model, "device": asr.device, "n_samples": len(items),
        "overall": overall,
        "per_language": {k: summarize(v) for k, v in per_lang.items()},
        "load_time_s": round(load_s, 2),
        "speedup_vs_realtime": round(1 / overall["rtf"], 1) if overall["rtf"] else None,
        "ram_total_mb": mb(proc.memory_info().rss),
        "gpu_memory_mb": gpu,
        "worst_samples": sorted(samples, key=lambda x: -x["wer"])[:10],
        "samples": samples,
    }

    print(f"WER {overall['wer']:.2%}  CER {overall['cer']:.2%}  RTF {overall['rtf']} "
          f"(~{result['speedup_vs_realtime']}x faster than real time)")
    for lang, v in result["per_language"].items():
        print(f"  {lang}: WER {v['wer']:.2%}  CER {v['cer']:.2%}  ({v['audio_s']}s audio)")
    print(f"  RAM {result['ram_total_mb']} MB, GPU memory {gpu} MB")
    print("  Worst examples:")
    for w in result["worst_samples"][:5]:
        if w["wer"] == 0:
            break
        print(f"    WER {w['wer']:.0%} [{w['lang']}]\n      ref: {w['reference']}\n      hyp: {w['hypothesis']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--librispeech", type=int, metavar="N")
    src.add_argument("--own", action="store_true")
    parser.add_argument("-m", "--models", nargs="+", default=[MODEL_NAME])
    parser.add_argument("--device")
    args = parser.parse_args()

    if args.own:
        source, items = "own_recordings", load_own()
    else:
        source, items = f"librispeech_{args.librispeech}", load_librispeech(args.librispeech)
    print(f"Dataset: {source}, {len(items)} utterances")

    results = [evaluate(m, items, args.device) for m in args.models]

    table = ["| Model | WER | CER | RTF | × real time | Load, s | GPU mem, MB |", "|---|---|---|---|---|---|---|"]
    for r in results:
        o = r["overall"]
        table.append(f"| {r['model']} | {o['wer']:.2%} | {o['cer']:.2%} | {o['rtf']} | "
                     f"{r['speedup_vs_realtime']} | {r['load_time_s']} | {r['gpu_memory_mb']} |")
    print("\n" + "\n".join(table))

    out = HERE / "results" / f"metrics_{source}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
