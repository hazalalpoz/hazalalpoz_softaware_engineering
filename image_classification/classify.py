import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import models

MODEL_NAME = "resnet50"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ImageClassifier:

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device or pick_device()
        self.weights = models.get_model_weights(model_name).DEFAULT
        self.model = models.get_model(model_name, weights=self.weights).eval().to(self.device)
        self.preprocess = self.weights.transforms()
        self.categories = self.weights.meta["categories"]

    @property
    def info(self) -> dict:
        m = self.weights.meta
        imagenet = m.get("_metrics", {}).get("ImageNet-1K", {})
        return {"weights": str(self.weights), "params_m": round(m.get("num_params", 0) / 1e6, 1),
                "gflops": m.get("_ops"), "file_mb": m.get("_file_size"),
                "imagenet_top1": imagenet.get("acc@1"), "imagenet_top5": imagenet.get("acc@5")}

    @torch.inference_mode()
    def logits(self, images: list[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(img.convert("RGB")) for img in images]).to(self.device)
        return self.model(batch).float().cpu()

    def predict(self, images, top_k: int = 5) -> list[list[dict]]:
        if not isinstance(images, (list, tuple)):
            images = [images]
        images = [Image.open(i) if isinstance(i, (str, Path)) else i for i in images]
        probs = self.logits(images).softmax(dim=1)
        top = probs.topk(top_k, dim=1)
        return [[{"label": self.categories[i], "class_id": int(i), "prob": round(float(p), 4)}
                 for p, i in zip(ps, ids)] for ps, ids in zip(top.values, top.indices)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Image classification (ImageNet, 1000 classes)")
    parser.add_argument("images", nargs="+")
    parser.add_argument("-m", "--model", default=MODEL_NAME)
    parser.add_argument("-k", "--top-k", type=int, default=5)
    parser.add_argument("--device")
    args = parser.parse_args()

    t0 = time.perf_counter()
    clf = ImageClassifier(args.model, args.device)
    print(f"{args.model} loaded on '{clf.device}' in {time.perf_counter() - t0:.1f}s  {clf.info}\n")

    for path in args.images:
        t0 = time.perf_counter()
        preds = clf.predict(path, args.top_k)[0]
        ms = (time.perf_counter() - t0) * 1000
        print(f"{path}  ({ms:.0f} ms)")
        for p in preds:
            print(f"  {p['prob']:6.1%}  {p['label']}")
        print()


if __name__ == "__main__":
    main()
