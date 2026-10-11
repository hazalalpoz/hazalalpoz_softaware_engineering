import argparse
import time

import torch
from transformers import pipeline

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SentimentAnalyzer:

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device or pick_device()
        self.pipe = pipeline("text-classification", model=model_name, device=self.device,
                              top_k=None, truncation=True, max_length=512)

    def predict(self, texts, batch_size: int = 16):
        single = isinstance(texts, str)
        out = self.pipe([texts] if single else list(texts), batch_size=batch_size)
        results = []
        for scores in out:
            scores = sorted(scores, key=lambda s: -s["score"])
            results.append({"label": scores[0]["label"].lower(), "score": round(scores[0]["score"], 4),
                             "scores": {s["label"].lower(): round(s["score"], 4) for s in scores}})
        return results[0] if single else results


def main() -> None:
    parser = argparse.ArgumentParser(description="Multilingual sentiment analysis (negative/neutral/positive)")
    parser.add_argument("texts", nargs="+")
    parser.add_argument("-m", "--model", default=MODEL_NAME)
    parser.add_argument("--device")
    args = parser.parse_args()

    t0 = time.perf_counter()
    sa = SentimentAnalyzer(args.model, args.device)
    print(f"{args.model} loaded on '{sa.device}' in {time.perf_counter() - t0:.1f}s\n")

    for text in args.texts:
        t0 = time.perf_counter()
        r = sa.predict(text)
        ms = (time.perf_counter() - t0) * 1000
        print(f"{text!r}  ({ms:.0f} ms)")
        print(f"  {r['label']}  {r['scores']}")
        print()


if __name__ == "__main__":
    main()
