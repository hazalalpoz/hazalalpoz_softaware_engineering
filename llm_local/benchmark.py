import argparse
import json
from pathlib import Path

import requests

from chat import HOST, check_ollama

HERE = Path(__file__).parent
DEFAULT_MODELS = ["llama3.2:3b", "llama3.1:8b"]

PROMPTS = [
    ("fact", "What is the capital of Australia? Answer with only the city name.", "canberra"),
    ("math", "A train travels 120 km in 1.5 hours. What is its average speed in km/h? "
             "Answer with only the number.", "80"),
    ("reasoning", "I have 3 apples, eat one and buy 5 more. How many apples do I have? "
                  "Answer with only the number.", "7"),
    ("code", "Write a Python function is_prime(n) that returns True if n is a prime number.", "def is_prime"),
    ("russian", "Переведи на английский: 'Я люблю программирование.'", "programming"),
    ("turkish", "Türkiye'nin başkenti neresidir? Sadece şehir adını yaz.", "ankara"),
    ("networking", "Which transport protocol does DNS primarily use, TCP or UDP? "
                    "Answer with only the protocol name.", "udp"),
    ("summary", "Summarize in one sentence: The water cycle moves water between the ocean, atmosphere "
                "and land through evaporation, condensation and precipitation.", None),
]


def generate(model: str, prompt: str) -> dict:
    r = requests.post(f"{HOST}/api/generate", json={
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_predict": 512},
    }, timeout=180)
    r.raise_for_status()
    d = r.json()
    gen_s = d.get("eval_duration", 0) / 1e9
    return {
        "response": d["response"].strip(),
        "load_s": round(d.get("load_duration", 0) / 1e9, 2),
        "prompt_eval_s": round(d.get("prompt_eval_duration", 0) / 1e9, 2),
        "eval_count": d.get("eval_count", 0),
        "tokens_per_s": round(d.get("eval_count", 0) / gen_s, 1) if gen_s else None,
    }


def memory_mb(model: str) -> float | None:
    r = requests.get(f"{HOST}/api/ps", timeout=10)
    r.raise_for_status()
    for m in r.json().get("models", []):
        if m["name"] == model:
            size = m.get("size_vram") or m.get("size")
            return round(size / 1024**2, 1) if size else None
    return None


def benchmark(model: str) -> dict:
    print(f"\n=== {model} ===")
    check_ollama(model)
    results = []
    for i, (name, prompt, keyword) in enumerate(PROMPTS):
        r = generate(model, prompt)
        if i == 0:
            print(f"Loaded on first request in {r['load_s']}s")
        ok = (keyword in r["response"].lower()) if keyword else None
        results.append({"prompt": name, "pass": ok, **r})
        tag = "PASS" if ok else "FAIL" if ok is False else "  - "
        print(f"  [{tag}] {name}: {r['response'][:60]!r}  ({r['tokens_per_s']} tok/s)")

    timed = [r for r in results if r["tokens_per_s"] and r["eval_count"] > 5]
    avg_tps = round(sum(r["tokens_per_s"] for r in timed) / len(timed), 1) if timed else None
    checked = [r for r in results if r["pass"] is not None]
    passed = sum(1 for r in checked if r["pass"])
    mem = memory_mb(model)

    summary = {
        "model": model,
        "load_s": results[0]["load_s"],
        "prompt_eval_s": round(sum(r["prompt_eval_s"] for r in results) / len(results), 2),
        "tokens_per_s_avg": avg_tps,
        "memory_mb": mem,
        "keyword_pass": f"{passed}/{len(checked)}",
        "prompts": results,
    }
    print(f"Load {summary['load_s']}s | Prompt eval {summary['prompt_eval_s']}s | "
          f"Generation ~{avg_tps} tok/s | Memory {mem} MB | Keyword checks {summary['keyword_pass']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    results = [benchmark(m) for m in args.models]

    table = ["| Model | Load, s | Prompt eval, s | Generation, tok/s (avg) | Memory, MB | Keyword checks |",
             "|---|---|---|---|---|---|"]
    for r in results:
        table.append(f"| {r['model']} | {r['load_s']} | {r['prompt_eval_s']} | {r['tokens_per_s_avg']} | "
                     f"{r['memory_mb']} | {r['keyword_pass']} |")
    print("\n" + "\n".join(table))

    out = HERE / "results" / "benchmark.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (out.parent / "benchmark.md").write_text("\n".join(table) + "\n")
    print(f"\nSaved to {out} and benchmark.md")


if __name__ == "__main__":
    main()
