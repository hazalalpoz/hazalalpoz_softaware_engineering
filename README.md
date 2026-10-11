# Applied ML with ready-made models

Solving applied problems with pretrained models only (no training, no fine-tuning). Everything runs locally on a MacBook Pro (Apple M3 Pro, 18 GB).

| # | Content type | Task | Framework | Model | Folder |
|---|---|---|---|---|---|
| 1 | Text | Multilingual sentiment analysis | Hugging Face Transformers | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | [`text_sentiment/`](text_sentiment) |
| 2 | Audio | Speech-to-text (EN / RU / TR) | Hugging Face Transformers | `openai/whisper-small` (+ `whisper-large-v3-turbo`) | [`audio_asr/`](audio_asr) |
| 3 | Image | Image classification (1000 ImageNet classes) | PyTorch / torchvision (PyTorch Hub) | `resnet50` (+ `mobilenet_v3_large`, `convnext_tiny`) | [`image_classification/`](image_classification) |
| 4 | Video | Object detection + tracking | Ultralytics YOLO (PyTorch) | `yolo26n` (+ `yolo26s`, `yolo26m`) | [`video_detection/`](video_detection) |
| 5 | LLM | Local chat assistant + benchmark | Ollama (llama.cpp, GGUF) | `llama3.1:8b` (+ `llama3.2:3b`) | [`llm_local/`](llm_local) |

Four different frameworks are used: Hugging Face Transformers, torchvision, Ultralytics and Ollama.

Full report: [REPORT.md](REPORT.md)

## Technical specification

**1. Text — sentiment analysis.** Input: a short text (review, comment, message) in English, Russian or Turkish. Output: `negative / neutral / positive` plus a probability for each class. Latency under ~100 ms per text; `SentimentAnalyzer.predict()` is ready to be wrapped in a REST API.

**2. Audio — speech recognition.** Input: an audio file (wav, flac, mp3, m4a from Voice Memos) in English, Russian or Turkish. Output: the transcript (optionally translated to English). Must run faster than real time (RTF < 1) and work offline after the first download.

**3. Image — classification.** Input: a photo. Output: top-5 ImageNet classes with probabilities. Compare a classic, a mobile and a modern architecture on accuracy vs. speed.

**4. Video — object detection.** Input: a video file or webcam stream. Output: an annotated video with boxes, class labels and track IDs, plus statistics (objects per frame, number of unique objects per class, FPS). Should run in real time (≥ 25–30 FPS) on a laptop.

**5. LLM — local assistant.** Deploy an open-source 3–8B instruction-tuned LLM with no external API calls. Interactive multi-turn chat and a benchmark of speed, memory and basic correctness.

## Setup (macOS, Apple Silicon)

```bash
git clone <repo-url> && cd ml-applied-tasks
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For the LLM, install Ollama (`brew install ollama` or the app from ollama.com), then `ollama pull llama3.1:8b`.

Models and datasets download automatically on first run (Hugging Face Hub, PyTorch Hub, Ultralytics).

## Usage

```bash
# 1. Text
cd text_sentiment
python sentiment.py "I love this!" "Ужасный сервис" "Fena değil"
python evaluate.py                       # hand-labelled EN/RU/TR set
python evaluate.py --tweet-eval 500      # 500 TweetEval tweets

# 2. Audio
cd audio_asr
python transcribe.py my_voice.m4a                 # language detected automatically
python transcribe.py my_voice.m4a -l turkish --translate
python evaluate.py --librispeech 100              # WER on LibriSpeech test-clean
python evaluate.py --own -m openai/whisper-small openai/whisper-large-v3-turbo   # your recordings

# 3. Image
cd image_classification
python classify.py photo.jpg
python evaluate.py -m resnet50 mobilenet_v3_large convnext_tiny   # Imagenette, 500 images

# 4. Video
cd video_detection
python detect_video.py street.mp4                 # -> results/annotated_street_yolo26n.mp4
python detect_video.py 0 --show                   # webcam, live window
python evaluate.py --video street.mp4             # mAP on COCO128 + FPS, yolo26 n/s/m

# 5. LLM
cd llm_local
python chat.py
python benchmark.py -m llama3.2:3b llama3.1:8b
```

Metrics are saved to `*/results/`. Downloaded data, weights and videos are git-ignored.
