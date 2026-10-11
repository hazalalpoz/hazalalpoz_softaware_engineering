import argparse
import json

import requests

MODEL_NAME = "llama3.1:8b"
HOST = "http://localhost:11434"


def check_ollama(model: str) -> None:
    try:
        r = requests.get(f"{HOST}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.RequestException:
        raise SystemExit("Ollama is not running. Start it with `ollama serve` (or open the Ollama app).")
    names = [m["name"] for m in r.json().get("models", [])]
    if model not in names:
        raise SystemExit(f"Model '{model}' is not downloaded. Run: ollama pull {model}\n"
                         f"Available locally: {names}")


class ChatSession:

    def __init__(self, model: str = MODEL_NAME, system: str | None = None):
        self.model = model
        self.history = [{"role": "system", "content": system}] if system else []
        self.last_stats: dict = {}

    def send(self, text: str):
        self.history.append({"role": "user", "content": text})
        r = requests.post(f"{HOST}/api/chat",
                          json={"model": self.model, "messages": self.history, "stream": True},
                          stream=True, timeout=180)
        r.raise_for_status()
        reply = ""
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            piece = chunk.get("message", {}).get("content", "")
            reply += piece
            yield piece
            if chunk.get("done"):
                gen_s = chunk.get("eval_duration", 0) / 1e9
                self.last_stats = {
                    "total_s": round(chunk.get("total_duration", 0) / 1e9, 2),
                    "load_s": round(chunk.get("load_duration", 0) / 1e9, 2),
                    "eval_tokens": chunk.get("eval_count"),
                    "tokens_per_s": round(chunk.get("eval_count", 0) / gen_s, 1) if gen_s else None,
                }
        self.history.append({"role": "assistant", "content": reply})

    def reset(self) -> None:
        self.history = [m for m in self.history if m["role"] == "system"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive chat with a local Ollama model")
    parser.add_argument("-m", "--model", default=MODEL_NAME)
    parser.add_argument("--system", help="system prompt")
    args = parser.parse_args()

    check_ollama(args.model)
    chat = ChatSession(args.model, args.system)
    print(f"Chatting with '{args.model}'. Commands: /reset (clear history), /exit (quit).\n")

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text == "/exit":
            break
        if text == "/reset":
            chat.reset()
            print("(history cleared)\n")
            continue

        print("assistant> ", end="", flush=True)
        for piece in chat.send(text):
            print(piece, end="", flush=True)
        tps = chat.last_stats.get("tokens_per_s")
        print(f"\n  ({tps} tok/s)\n" if tps else "\n")


if __name__ == "__main__":
    main()