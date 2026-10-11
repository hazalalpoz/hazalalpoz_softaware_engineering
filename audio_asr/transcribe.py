import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import pipeline

MODEL_NAME = "openai/whisper-small"
SAMPLE_RATE = 16_000


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _to_wav_with_afconvert(path: Path) -> Path:
    if not shutil.which("afconvert"):
        raise RuntimeError(f"Cannot read {path.suffix} files. Convert to .wav first "
                           "(e.g. with ffmpeg) or run on macOS.")
    out = Path(tempfile.mkdtemp()) / (path.stem + ".wav")
    r = subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SAMPLE_RATE}", "-c", "1",
                        str(path), str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Could not convert {path}: {r.stderr.strip() or 'afconvert failed'}")
    return out


def load_audio(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        similar = sorted(p.name for p in path.parent.glob(path.stem + ".*")) if path.parent.exists() else []
        raise SystemExit(f"Audio file not found: {path}" + (f"  (did you mean: {', '.join(similar)}?)" if similar else ""))
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        audio, sr = sf.read(_to_wav_with_afconvert(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        g = np.gcd(sr, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)
    return audio


class SpeechRecognizer:

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device or pick_device()
        self.pipe = pipeline("automatic-speech-recognition", model=model_name, device=self.device)

    def transcribe(self, audio: np.ndarray, language: str | None = None,
                   translate: bool = False) -> dict:
        gen = {"task": "translate" if translate else "transcribe"}
        if language:
            gen["language"] = language 
        duration = len(audio) / SAMPLE_RATE
        extra = {"return_timestamps": True} if duration > 30 else {}
        t0 = time.perf_counter()
        out = self.pipe({"raw": audio, "sampling_rate": SAMPLE_RATE}, generate_kwargs=gen, **extra)
        elapsed = time.perf_counter() - t0
        return {
            "text": out["text"].strip(),
            "audio_s": round(duration, 2),
            "processing_s": round(elapsed, 2),
            "rtf": round(elapsed / duration, 3) if duration else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Speech-to-text with Whisper")
    parser.add_argument("files", nargs="+", help="audio file(s): wav, flac, mp3, m4a ...")
    parser.add_argument("-m", "--model", default=MODEL_NAME)
    parser.add_argument("-l", "--language", help="e.g. english, russian, turkish (auto if omitted)")
    parser.add_argument("--translate", action="store_true", help="translate speech to English")
    parser.add_argument("--device", help="cpu / mps / cuda")
    args = parser.parse_args()

    t0 = time.perf_counter()
    asr = SpeechRecognizer(args.model, args.device)
    print(f"{args.model} loaded on '{asr.device}' in {time.perf_counter() - t0:.1f}s\n")

    for f in args.files:
        r = asr.transcribe(load_audio(f), args.language, args.translate)
        print(f"[{f}] {r['audio_s']}s audio -> {r['processing_s']}s (RTF {r['rtf']})")
        print(f"  {r['text']}\n")


if __name__ == "__main__":
    main()
