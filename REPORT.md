# Report

Hardware: MacBook Pro, Apple M3 Pro, 18 GB unified memory, macOS 15.1.1. Software: Python 3.14.8, PyTorch 2.14.1, Ultralytics 8.4.174, Ollama 0.35.1.

> All numbers below were measured on this machine with each task's `evaluate.py` / `benchmark.py`; raw results are in the `results/` folders.

---

## 1. Text processing — multilingual sentiment analysis

### 1.1 Library and its advantages

**Hugging Face Transformers** (PyTorch backend).

- Hub with hundreds of thousands of pretrained models; a model is loaded by its name in one line.
- The `pipeline` API hides tokenization, batching, the forward pass and post-processing (softmax, label mapping).
- Same code runs on CPU, NVIDIA (CUDA) and Apple Silicon (MPS) — only the `device` argument changes.
- Models are cached locally after the first download (`~/.cache/huggingface`), so inference is fully offline afterwards.
- Large community, model cards with documented training data and limitations.

### 1.2 Model: how it works and why it was chosen

**`cardiffnlp/twitter-xlm-roberta-base-sentiment`** (XLM-T, Barbieri et al., 2022).

How it works:
1. **Tokenization** — the text is split into sub-word pieces with a SentencePiece vocabulary shared by ~100 languages, so Russian, Turkish and English words map into the same vocabulary.
2. **Encoder** — XLM-RoBERTa-base: a 12-layer Transformer encoder (~278M parameters). Self-attention lets every token look at every other token, so the representation of "bad" in "not bad" is affected by "not".
3. **Pretraining** — XLM-R was pretrained with masked language modelling (predict hidden words) on 2.5 TB of CommonCrawl text in 100 languages; XLM-T continued this pretraining on ~198M multilingual tweets.
4. **Classification head** — the vector of the first special token `<s>` goes through a small dense layer and softmax over 3 classes: negative / neutral / positive. This head was fine-tuned by the authors on tweet sentiment data in 8 languages.

Why this model:
- **Multilingual**: one model for English, Russian and Turkish. Thanks to cross-lingual transfer, it handles languages it wasn't sentiment-tuned on (e.g. Russian, Turkish) because they share the representation space.
- **Three classes including neutral** — more realistic than binary positive/negative (e.g. SST-2 models).
- **Trained on informal text** (tweets): slang, emojis, short messages — closer to real reviews/comments.
- **Base size**: fast enough for real-time use on a laptop and a good candidate for the REST API part.

Alternatives considered: `distilbert-base-uncased-finetuned-sst-2-english` (faster but English-only, binary), `blanchefort/rubert-base-cased-sentiment` (Russian-only).

### 1.3 Dataset structure

**Training (done by the model authors, not by us):** multilingual tweets, each with one label `{negative, neutral, positive}`.

**Inference / evaluation (ours):**

| File | Columns | Size |
|---|---|---|
| `text_sentiment/data/test_samples.csv` | `text` (string), `label` (negative/neutral/positive), `lang` (en/ru/tr) | 30 rows, 10 per language; 11 positive / 10 negative / 9 neutral |
| TweetEval `sentiment` test split (optional, `--tweet-eval N`) | `text`, `label` (0/1/2) | 12,284 tweets; random sample of N |

The local set intentionally includes hard cases: negation ("Not bad at all", "Fena değil") and sarcasm ("Ну да, конечно, именно этого я и ждал...").

Model input: raw string, truncated to 512 tokens. Output: `{"label": ..., "score": ..., "scores": {negative, neutral, positive}}`.

### 1.4 Quality metrics

ML metrics (`evaluate.py`):
- **Accuracy** — share of correct predictions.
- **Precision / Recall / F1 per class** and **macro F1** — macro F1 matters because a model can have good accuracy while failing on one class (usually *neutral*).
- **Confusion matrix** — shows which classes get confused.
- **Per-language accuracy** — checks that the model isn't only good at English.
- **Error list** — misclassified texts for manual analysis.

Infrastructure metrics: model load time, latency per text (p50/p95), batch throughput, RAM and GPU (MPS) memory.

Product metrics (for the future API): requests per second, error rate, share of low-confidence predictions (max probability < 0.5), which could be routed to a human.

Results:

| Dataset | Accuracy | Macro F1 | EN / RU / TR accuracy |
|---|---|---|---|
| Local (30) | 0.933 | 0.930 | 1.00 / 0.90 / 0.90 |
| TweetEval (500 random test tweets, EN) | 0.670 | 0.672 | — |

| Device | Load, s | Latency p50 / p95, ms | Throughput, texts/s (batch 16) | Memory, MB |
|---|---|---|---|---|
| MPS (M3 Pro GPU), local set | 2.7 | 11.8 / 15.5 | 532.2 | 1099 (GPU) |
| MPS (M3 Pro GPU), TweetEval 500 | 2.1 | 9.3 / 12.5 | 296.1 | 2156 (GPU) |
| CPU (M3 Pro), local set | 2.0 | 27.5 / 30.1 | 262.2 | ~323 (RAM, see note) |

Per-class results on the local set: positive — precision/recall 1.00; negative — 0.90/0.90; neutral — 0.89/0.89.
Both negation cases ("Not bad at all", "Fena değil") were classified correctly as positive, which shows the model uses context, not just individual words.

Only 2 of 30 texts were misclassified:

| Text | True | Predicted | Why |
|---|---|---|---|
| Ну да, конечно, именно этого я и ждал... | negative | neutral | Sarcasm: the literal words are neutral/positive; the negative meaning comes from tone and situation, which the model cannot see. |
| Kütüphane hafta sonu kapalıdır. | neutral | negative | A factual statement, but the word *kapalı* ("closed") often appears in complaints in tweets, so the model links it with negative sentiment. The label itself is also debatable — for a reader this can be mildly bad news. |

**Results on real data (TweetEval, 500 tweets):**

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.62 | 0.87 | 0.73 | 163 |
| neutral | 0.72 | 0.52 | 0.60 | 223 |
| positive | 0.70 | 0.68 | 0.69 | 114 |

Confusion matrix (rows = true, columns = predicted):

| | negative | neutral | positive |
|---|---|---|---|
| negative | 142 | 18 | 3 |
| neutral | 77 | 116 | 30 |
| positive | 9 | 28 | 77 |

TweetEval's official metric is macro-averaged recall, which here is (0.87 + 0.52 + 0.68) / 3 ≈ 0.69.

Analysis:
- **Big gap between the two test sets (0.93 vs 0.67).** Our hand-written sentences are short and unambiguous; real tweets contain sarcasm, slang, hashtags, mentions, politics and missing context. A small self-made test set overestimates quality — evaluation must be done on real data.
- **The main weakness is the neutral class.** 77 of 223 neutral tweets (35%) were predicted as negative. Neutral news or statements that contain negative-sounding words ("RIP", "traumatic", "rubber stamp") get pulled toward negative. Because of this, negative has high recall (0.87) but low precision (0.62): the model is biased toward the negative class.
- **Positive and negative are almost never confused with each other** (only 3 + 9 cases). Most mistakes are on the neutral border, which is also where human annotators disagree most.
- **Some gold labels are debatable** — e.g. a tweet about a "really traumatic" Valentine's experience is labelled neutral, while a hostile reply without emotional words is labelled negative. Label noise limits the maximum achievable score.
- Possible improvements without training: an English-specific model for English-only data (e.g. `cardiffnlp/twitter-roberta-base-sentiment-latest`), or treating low-confidence predictions (max probability < 0.5) as "uncertain".

Efficiency notes:
- **GPU vs CPU:** predictions were identical on both devices (same accuracy, same 2 errors) — the device affects only speed. MPS is ~2.3× faster per text (11.8 vs 27.5 ms) and ~2× faster in batch mode (532 vs 262 texts/s). Even on CPU the latency stays well under the 100 ms target, so the API could also run on a server without a GPU.
- On CPU the measured RAM increase (~323 MB) is lower than the real model size (~1.1 GB) because the weights are loaded from a memory-mapped safetensors file, and macOS does not count all file-backed pages in the process RSS.
- Latency of ~12 ms per text is far below the 100 ms target from the specification, so the model is suitable for a real-time API.
- Batching gives ~532 texts/s vs ~85 texts/s one-by-one (1000 / 11.8 ms) — about 6× higher throughput.
- GPU memory (~1.1 GB) matches the model size: ~278M parameters × 4 bytes (fp32).
- Process RAM barely changes on MPS (the measured difference was ≈ 0, −15 MB) because the weights are moved to GPU memory after loading, so on MPS the GPU memory value is the meaningful one.
- On TweetEval, batch throughput is lower (296 vs 532 texts/s) because tweets are longer: every batch is padded to its longest text, so there is more computation per batch. GPU memory is also higher (2.2 GB) because the MPS allocator caches buffers for the larger tensors — it is reserved memory, not model size.
- Load time of 2.7 s is with the model already cached locally; the first run also downloads ~1.1 GB from the Hugging Face Hub.

### 1.5 Implementation details and efficiency

- `SentimentAnalyzer` loads the model **once**; `predict()` accepts one text or a list — this is the function a future FastAPI endpoint will call.
- **Device auto-selection**: MPS → CUDA → CPU.
- **`top_k=None`** returns probabilities for all classes, not only the top one — useful for confidence thresholds.
- **Batching** (`batch_size=16`) gives much higher throughput than one-by-one calls.
- **Warm-up** call before timing: the first MPS call compiles GPU kernels and would distort latency.
- **Truncation** protects against texts longer than 512 tokens.
- Limitations: sarcasm, domain shift (tweets vs. formal text), no sentiment fine-tuning on Russian/Turkish specifically.

---

## 2. Audio processing — speech-to-text (speech recognition)

### 2.1 Library and its advantages

**Hugging Face Transformers** — the `automatic-speech-recognition` pipeline.

- Whisper checkpoints of every size are on the Hub; switching model = changing one string.
- The pipeline does feature extraction (log-Mel spectrogram), generation and decoding of tokens into text.
- Built-in support for language selection, translation to English and long audio (timestamp-based long-form decoding).
- Runs on Apple Silicon GPU (MPS); cached locally after the first download.
- Audio is read with `soundfile` and resampled with `scipy`; `.m4a` files from Voice Memos are converted with the macOS built-in `afconvert`, so no ffmpeg is needed.

### 2.2 Model: how it works and why it was chosen

**OpenAI Whisper** (Radford et al., 2022): `openai/whisper-small` (244M parameters) as the main model, `openai/whisper-large-v3-turbo` (809M) for comparison.

How it works:
1. **Pre-processing**: audio → mono 16 kHz → **log-Mel spectrogram** (a "picture" of the sound: time × frequency, on the Mel scale that matches human hearing). Whisper processes 30-second windows.
2. **Encoder** (Transformer): two convolution layers downsample the spectrogram, then self-attention layers build a representation of the whole 30 s window.
3. **Decoder** (Transformer): generates text **token by token** (like an LLM), attending to the encoder output (cross-attention). Special tokens at the start tell it the **language** (`<|ru|>`, `<|tr|>` …) and the **task** (`<|transcribe|>` or `<|translate|>`). If the language isn't given, the model predicts the language token itself.
4. **Training**: weak supervision on a huge set of audio with transcripts from the internet (680k hours for the original models; large-v3 uses about 5M hours). This is why it is robust to accents, noise and different microphones without fine-tuning.
5. **Turbo**: large-v3 with the decoder reduced from 32 to 4 layers. Decoding is the slow, sequential part, so this makes it several times faster with a small loss in accuracy.

Why this model:
- **One multilingual model** for English, Russian and Turkish, plus speech translation.
- Robust "out of the box" — designed for zero-shot use, which matches the task (no training).
- Several sizes → a clear speed/accuracy trade-off to study; `small` is fast enough for near-real-time use, `turbo` is close to the best open-source quality.

Alternatives: Wav2Vec2 (needs a separate fine-tuned model per language, no punctuation), Vosk (lighter but less accurate).

### 2.3 Dataset structure

**Training** (done by OpenAI): pairs (audio segment, transcript) in ~99 languages.

**Evaluation (ours):**

| Dataset | Structure | Size |
|---|---|---|
| LibriSpeech `test-clean` (via Hugging Face datasets-server) | `id`, `audio` (FLAC, 16 kHz), `text` (upper-case transcript, no punctuation) | 2,620 utterances; random sample of 100 (seed 42, 771.7 s of audio), cached in `data/librispeech/` |
| Own recordings, `data/recordings.csv` | `file` (audio in `data/recordings/`), `lang` (en/ru/tr), `text` (reference transcript) | 15 sentences, 5 per language, recorded by the author (92.5 s of audio: EN 30.9 s, RU 35.2 s, TR 26.4 s) |

The own set includes everyday sentences and technical terms ("router", "маршрутизатор", "yönlendirici") to check domain vocabulary.

Model input: float32 waveform at 16 kHz (any length). Output: text + processing time.

### 2.4 Quality metrics

ML metrics (`evaluate.py`):
- **WER (word error rate)** = (substitutions + deletions + insertions) / number of words in the reference. The standard ASR metric; lower is better, 0% is perfect.
- **CER (character error rate)** — the same at character level. Fairer for morphologically rich languages (Turkish, Russian), where one wrong suffix makes the whole word wrong.
- Before comparing, both texts are **normalized** (lower-case, no punctuation; for English, Whisper's own normalizer also unifies numbers and spelling, e.g. "colour"/"color"), otherwise formatting differences would count as errors.
- **Per-language WER/CER** and the **worst examples** for error analysis.

Infrastructure metrics:
- **RTF (real-time factor)** = processing time / audio duration. RTF 0.1 means 10 s of audio is transcribed in 1 s; RTF < 1 is required for live use.
- Load time, GPU memory.

Results:

| Dataset | Model | WER | CER | RTF | × real time | GPU memory, MB |
|---|---|---|---|---|---|---|
| LibriSpeech (100 utterances) | whisper-small | 2.86% | 1.03% | 0.150 | 6.7 | 2522 |
| Own recordings (15) | whisper-small | 13.76% | 3.79% | 0.115 | 8.7 | 2476 |
| Own recordings (15) | whisper-large-v3-turbo | 6.42% | 2.31% | 0.153 | 6.5 | 2474 |

Per language (own recordings), WER / CER:

| Model | English | Russian | Turkish |
|---|---|---|---|
| whisper-small | 0.00% / 0.00% | 33.33% / 8.29% | 12.90% / 3.61% |
| whisper-large-v3-turbo | 0.00% / 0.00% | 18.18% / 5.18% | 3.23% / 2.06% |

Load time with the model already cached: 1.8 s (small). The first run includes the download: 68 s for small (~1 GB) and 157 s for turbo (~1.6 GB).

Errors on LibriSpeech (whisper-small):

| Reference | Recognized | Type of error |
|---|---|---|
| STEPHANOS DEDALOS | Stefano Steadloss | rare proper name |
| CAPTAIN DONNITHORNE | Captain Donnie Thorne | rare proper name split into common words |
| IF I CAN GET PATIENTS | If I can get patience? | homophone — the same sound, context needed |
| … PRETENDED SEARCH … MADE TWO ARRESTS | … pretended surge … made to arrest | similar-sounding words |

Errors on own recordings:

| Lang | Reference | small | turbo | Comment |
|---|---|---|---|---|
| RU | Маршрутизатор передаёт пакеты … | Маршрут и затор придают пакеты … | same as small | rare technical word split into two frequent words that sound the same ("route" + "traffic jam") |
| RU | Сегодня в Казани холодно но солнечно | Сегодня выказание холодное на солнечно | Сегодня в Казани холодно на солнечно | word boundaries lost (small); "но" → "на": unstressed "о" is pronounced like "а" in Russian |
| RU | … отправьте мне отчёт … | … отправьте меня отчет … | correct | wrong grammatical form |
| RU | … в десять часов | correct | … в 10 часов | not a real error: the number is written as digits, but the normalizer converts numbers only for English |
| TR | Bugün Kazan'da hava soğuk ama güneşli | Bugün kazandı hava soğuk ama güneşle | Bugün kazanda hava soğuk ama güneşli | "kazandı" = "won" (a real word that sounds similar); turbo only misses the apostrophe and capital letter, which is not counted as an error |
| TR | Makine öğrenmesi dersi saat onda başlıyor | … derse onda başlıyor | … dersi onda başlıyor | both models drop "saat"; small also changes the suffix |

Analysis:
- **English is almost perfect, other languages are harder.** On LibriSpeech, whisper-small reaches 2.86% WER (the Whisper paper reports about 3.4% for this model on the full test-clean set). On our English recordings both models made no errors. Russian and Turkish have much less training data than English, and all recordings come from one speaker on a consumer microphone, so accent and recording quality also play a role.
- **The bigger model halves the error rate:** 13.8% → 6.4% WER overall, Russian 33% → 18%, Turkish 12.9% → 3.2%. It fixes word-boundary and grammar errors. Both models still fail on the rare technical word "маршрутизатор": when the sound is ambiguous, the decoder (which works like a language model) prefers frequent words.
- **Turbo is only ~33% slower (RTF 0.115 → 0.153) despite having 3.3× more parameters.** Generation is the slow, sequential part, and turbo has only 4 decoder layers (small has 12); the large encoder runs once per 30 s window and is fast on the GPU. Both models are 6–9× faster than real time, so live transcription is possible.
- **CER is much lower than WER** (Turkish: 3.6% vs 12.9% for small). In agglutinative Turkish and inflected Russian one wrong suffix ("dersi" → "derse", "мне" → "меня") makes the whole word wrong, while only one or two characters differ. CER is therefore the fairer metric for these languages.
- **Normalization strongly affects WER.** In the first version of the evaluation, apostrophes were turned into spaces, so "Kazan'da" became two words and turbo's correct "kazanda" counted as two errors (Turkish WER 9.4% instead of 3.2%). After the fix apostrophes are removed. Likewise "десять" vs "10" is still counted as an error. Before comparing WER numbers, it is important to know how the text was normalized.
- **GPU memory (~2.5 GB) is the same for both models**, because `driver_allocated_memory` shows memory reserved by the MPS allocator, not the size of the model. The weight files are ~1 GB (small) and ~1.6 GB (turbo).
- Warnings printed by transformers (`generation_config` deprecation, `clean_up_tokenization_spaces`) do not affect the results.

### 2.5 Implementation details and efficiency

- `SpeechRecognizer` loads the model once; `transcribe()` accepts a waveform → ready for an API endpoint (e.g. file upload → text).
- Universal `load_audio()`: any format → mono → 16 kHz (polyphase resampling).
- Audio longer than 30 s automatically switches to Whisper's sequential long-form decoding (with timestamps), so lectures and long voice messages work too.
- Setting the language explicitly (`-l russian`) skips language detection and avoids wrong-language output on short clips.
- A warm-up call before measurements (first MPS call compiles kernels).
- Limitations: Whisper can "hallucinate" text on silence or music; proper nouns and rare terms are sometimes misspelled.

---

## 3. Image processing — image classification

### 3.1 Library and its advantages

**PyTorch / torchvision** (the models published on PyTorch Hub).

- Dozens of pretrained architectures behind one API: `get_model(name, weights="DEFAULT")`.
- Each weights object carries its **own pre-processing** (`weights.transforms()`), class names and official metrics, so inference is consistent with training and no "magic numbers" are needed.
- Native PyTorch: runs on MPS, easy batching, easy export (ONNX, Core ML) for deployment.
- Different framework from tasks 1–2 (no Transformers pipeline), which satisfies the "at least 2 frameworks" requirement.

### 3.2 Model: how it works and why it was chosen

Main model: **ResNet-50** (He et al., 2015), torchvision `IMAGENET1K_V2` weights. Compared with **MobileNetV3-Large** and **ConvNeXt-Tiny**.

How a CNN classifier works:
1. **Pre-processing**: resize to 232 px, center-crop 224 × 224, normalize with ImageNet mean/std.
2. **Convolutional layers** slide small filters over the image. Early layers detect edges and colors, middle layers textures and parts (eyes, wheels), deep layers whole objects.
3. **Residual connections** (ResNet's key idea): each block outputs `x + F(x)`, so it only has to learn a correction. This lets gradients flow through 50 layers during training and made very deep networks trainable.
4. **Global average pooling** → a 2048-number feature vector → **fully connected layer** → 1000 scores → **softmax** = probabilities of the 1000 ImageNet classes.

The other two:
- **MobileNetV3** — depthwise-separable convolutions (filtering each channel separately, then mixing), squeeze-and-excitation attention, architecture found by neural architecture search; ~19× fewer operations than ResNet-50, designed for phones.
- **ConvNeXt** — a ResNet "modernized" with ideas from Vision Transformers: large 7×7 depthwise kernels, LayerNorm, GELU, inverted bottlenecks. Higher accuracy with similar cost.

| Model | Params | GFLOPs | Official ImageNet top-1 / top-5 |
|---|---|---|---|
| resnet50 (V2) | 25.6M | 4.09 | 80.9% / 95.4% |
| mobilenet_v3_large (V2) | 5.5M | 0.22 | 75.3% / 92.6% |
| convnext_tiny | 28.6M | 4.46 | 82.5% / 96.1% |

Why ResNet-50: the reference architecture of computer vision — well understood, strong accuracy, and the V2 weights (trained with a modern recipe) make it competitive. The comparison shows the classic trade-off: mobile (fast) vs. classic vs. modern (accurate).

### 3.3 Dataset structure

**Training** (by torchvision): **ImageNet-1K** — 1.28M training images, 1000 classes (dog breeds, vehicles, objects …), each image with one label.

**Evaluation (ours): Imagenette** (fast.ai) — a 10-class subset of ImageNet; images are JPEGs in folders named by WordNet ID (`val/n01440764/*.JPEG` = tench). Downloaded automatically (~330 MB, `imagenette2-320`), `--per-class` images sampled per class (default 50 → 500 images, seed 42). Each folder is mapped to its ImageNet class index (e.g. `n02102040` → 217 "English springer").

Model input: RGB image (any size) → tensor 3 × 224 × 224. Output: 1000 probabilities → top-k labels.

### 3.4 Quality metrics

ML metrics:
- **Top-1 accuracy (1000 classes)** — the predicted class is exactly right. Strict: the model may choose among all 1000 classes, including very similar ones.
- **Top-5 accuracy** — the correct class is among the 5 most probable. Fairer when classes are near-duplicates (e.g. dog breeds).
- **10-class accuracy** — the same model restricted to the 10 Imagenette classes (argmax over their 10 scores). Shows how much error comes from confusing similar fine-grained classes vs. truly not recognizing the object.
- **Per-class accuracy** and **most frequent confusions** (true → predicted).

Infrastructure metrics: latency per image (p50/p95), throughput in batches (img/s), load time, memory, parameters, GFLOPs.

Results (500 images, 50 per class, MPS, batch size 32):

| Model | Top-1 (1000) | Top-5 | 10-class acc. | Official ImageNet top-1 | Latency p50 / p95, ms | Throughput, img/s |
|---|---|---|---|---|---|---|
| resnet50 | 86.8% | 99.4% | 99.8% | 80.9% | 8.3 / 9.3 | 181.6 |
| mobilenet_v3_large | 80.6% | 97.2% | 99.6% | 75.3% | 6.7 / 8.5 | 359.6 |
| convnext_tiny | 88.4% | 99.6% | 100.0% | 82.5% | 7.8 / 9.1 | 152.7 |

Per-class top-1 accuracy:

| Class | resnet50 | mobilenet_v3_large | convnext_tiny |
|---|---|---|---|
| tench | 100% | 98% | 100% |
| English springer | 94% | 90% | 92% |
| cassette player | 56% | 42% | 56% |
| chain saw | 88% | 76% | 96% |
| church | 48% | 50% | 46% |
| French horn | 96% | 88% | 98% |
| garbage truck | 96% | 92% | 100% |
| gas pump | 98% | 90% | 100% |
| golf ball | 98% | 92% | 98% |
| parachute | 94% | 88% | 98% |

Most frequent confusions (true → predicted, number of images):

| resnet50 | mobilenet_v3_large | convnext_tiny |
|---|---|---|
| cassette player → tape player (10) | cassette player → tape player (15) | cassette player → tape player (15) |
| church → vault (7) | church → vault (7) | church → vault (7) |
| church → bell cote (7) | church → altar (5) | church → bell cote (6) |
| church → monastery (6) | church → monastery (5) | church → altar (5) |
| cassette player → cassette (5) | chain saw → lumbermill (3) | church → monastery (5) |

Analysis:
- **The ranking matches the official benchmark:** ConvNeXt-Tiny > ResNet-50 > MobileNetV3, with about the same gaps (88.4 / 86.8 / 80.6% here vs 82.5 / 80.9 / 75.3% on ImageNet). Absolute numbers are ~6 points higher because Imagenette contains 10 well-separated, common classes, while the full benchmark averages over all 1000 classes, including many hard ones.
- **Almost all "errors" are fine-grained confusions, not failures to recognize the object.** When the model only has to choose among the 10 Imagenette classes, accuracy is 99.6–100%, and top-5 accuracy is 97–99.6%. The mistakes are neighbouring ImageNet classes: a cassette player is predicted as "tape player" or "cassette"; a church photo as "vault", "bell cote", "altar" or "monastery" — often a part of a church or a very similar building. Many of these answers are reasonable for the picture; the problem is that one image has one label while it may show several valid things. This explains why church (46–50%) and cassette player (42–56%) are the weakest classes for all three models.
- **Top-1 alone underestimates the model.** For a real application the label set should be chosen for the task (e.g. merge similar classes or restrict to the needed ones, as in the 10-class metric), or top-5 should be shown to the user.
- **Latency is almost the same for all models (6.7–8.3 ms), although MobileNet needs ~19× fewer operations than ResNet-50.** For a single small image the time is dominated by fixed overheads: CPU pre-processing (resize, crop, normalize), copying to the GPU and launching GPU kernels, not by the computation itself.
- **In batches the difference appears:** MobileNet processes 360 img/s, about 2× ResNet-50 (182) and 2.4× ConvNeXt (153). It is still far from 19×, because the throughput includes CPU pre-processing of every image, and depthwise convolutions use the GPU less efficiently than ordinary convolutions — FLOPs alone do not predict speed.
- **Trade-off:** ConvNeXt-Tiny gives +1.6 points over ResNet-50 for ~16% lower throughput — the best choice when accuracy matters. MobileNetV3 loses ~6 points but is 2× faster with 5× fewer parameters (21 MB file) — the choice for phones, CPUs or high request volumes. ResNet-50 is a well-understood middle ground.
- **Memory:** GPU memory (1.2 GB for ResNet/MobileNet, 2.2 GB for ConvNeXt) is mostly memory reserved by the MPS allocator for activations of a 32-image batch, not the weights (21–109 MB files). Loading a cached model takes 0.1–0.4 s; the first run includes the download (~10 s).

### 3.5 Implementation details and efficiency

- `ImageClassifier` loads the model once (`eval()` mode, `torch.inference_mode()` — no gradient tracking, less memory, faster).
- Pre-processing comes from the weights object, so it always matches what the model was trained with.
- `predict()` accepts file paths or PIL images, single or batch → ready for an API.
- Batching (32 images) for throughput; single-image calls for latency; warm-up before timing.
- Limitations: only 1000 fixed ImageNet classes (no "person", for example); one label per image even when several objects are present — for that, detection (task 4) is needed.

---

## 4. Video processing — object detection and tracking

### 4.1 Library and its advantages

**Ultralytics** (YOLO family, built on PyTorch).

- One package for detection, segmentation, pose, classification and **tracking**; weights download automatically.
- Reads video files, webcams and streams directly (`source=...`), returns results frame by frame (`stream=True` → constant memory).
- Built-in trackers (ByteTrack, BoT-SORT), visualization (`result.plot()`), validation with standard COCO metrics (`model.val()`), export to Core ML / ONNX / TensorRT.
- Runs on Apple GPU (MPS).

### 4.2 Model: how it works and why it was chosen

**YOLO26** (Ultralytics, January 2026) — the newest released YOLO generation; `yolo26n` (nano) as the main model, `s` and `m` for comparison.

How it works:
1. Each frame is resized/padded to 640 × 640.
2. **Backbone** (convolutional network) extracts features; **neck** combines features from several scales (feature pyramid), so small and large objects are both visible.
3. **Head** predicts, in **one pass** ("You Only Look Once"), for every location on 3 grids: a bounding box (x, y, w, h) and scores for the 80 COCO classes.
4. **Post-processing**: low-confidence boxes are dropped (`conf` threshold); overlapping duplicates are removed with **NMS** (non-maximum suppression). YOLO26 also has an NMS-free one-to-one head (`nms=False`) for simpler, faster deployment. It removes DFL (distribution focal loss) to make the head lighter.
5. **Tracking (ByteTrack)**: a Kalman filter predicts where each object moves in the next frame; new detections are matched to existing tracks by box overlap (IoU) with the Hungarian algorithm. Low-confidence detections are used in a second matching round, so partially hidden objects keep their ID. Result: every object gets a stable **ID** → we can count unique objects instead of detections.

Official COCO val2017 results (Ultralytics):

| Model | Params | mAP50-95 | CPU ONNX, ms | T4 TensorRT, ms |
|---|---|---|---|---|
| yolo26n | 2.4M | 40.9 | 38.9 | 1.7 |
| yolo26s | 9.5M | 48.6 | 87.2 | 2.5 |
| yolo26m | 20.4M | 53.1 | 220.0 | 4.7 |

Why YOLO26: real-time one-stage detector (video needs ≥ 25–30 FPS), state-of-the-art accuracy/speed, 80 everyday classes (people, vehicles, animals, objects), built-in tracking. The nano model is fast enough for live webcam processing on a laptop.

Alternatives: DETR / RT-DETR (transformer detectors — accurate but heavier), Faster R-CNN (two-stage, slower).

### 4.3 Dataset structure

**Training** (by Ultralytics): **COCO** — ~118k training images, 80 classes, each object annotated with a bounding box and class.

**Evaluation (ours):**
- **COCO128** — 128 COCO images with YOLO-format labels (one `.txt` per image, each line `class x_center y_center width height`, normalized 0–1). Downloaded automatically (~7 MB). ⚠ These images come from COCO *train2017*, which the models were trained on, so the measured mAP is optimistic; it is a correctness/sanity check, the official val2017 numbers above are the real quality reference.
- **Own video** (`street.mp4`, a street scene recorded on a phone, 555 frames; the model finds mainly cars, people and motorcycles) — no labels; used for speed, objects per frame, unique object counts and visual inspection.

Model input: video frames (BGR images). Output per frame: boxes, class IDs, confidences, track IDs → annotated video + statistics JSON.

### 4.4 Quality metrics

ML metrics (`evaluate.py`, COCO128):
- **IoU** (intersection over union) — overlap between a predicted and a true box; a detection is correct if IoU ≥ threshold and the class matches.
- **Precision** (share of predicted boxes that are correct) and **Recall** (share of real objects that were found).
- **AP** — area under the precision–recall curve for one class; **mAP** — mean over all classes.
- **mAP50** — at IoU 0.5; **mAP50-95** — averaged over IoU 0.5, 0.55 … 0.95 (also rewards precise box placement; the main COCO metric).

Product / infrastructure metrics (own video):
- **FPS** — model-only (pre-process + inference + post-process) and end-to-end (including video decoding, drawing, writing). ≥ 25–30 FPS = real time.
- **ms per frame** per stage, average objects per frame, unique tracked objects per class, mean confidence.

Results:

| Model | Params (checkpoint / fused) | mAP50 | mAP50-95 | Precision | Recall | Pipeline, ms/frame | FPS (model) | FPS (end-to-end) |
|---|---|---|---|---|---|---|---|---|
| yolo26n | 2.6M / 2.4M | 0.646 | 0.484 | 0.734 | 0.549 | 8.0 | 125.2 | 79.6 |
| yolo26s | 10.0M / 9.5M | 0.751 | 0.570 | 0.794 | 0.655 | 10.9 | 92.1 | 63.8 |
| yolo26m | 21.9M / 20.4M | 0.778 | 0.617 | 0.765 | 0.683 | 22.6 | 44.3 | 36.5 |

mAP, precision and recall: COCO128 (128 images, 929 objects). FPS: first 300 frames of `street.mp4`, with tracking, without writing a video. MPS (M3 Pro GPU).

Objects found in the first 300 frames:

| Model | Objects per frame | Unique tracked objects |
|---|---|---|
| yolo26n | 4.65 | car 15, person 11, potted plant 5, motorcycle 3, chair 3, clock 1, bench 1 |
| yolo26s | 7.01 | car 20, person 13, chair 7, motorcycle 6, umbrella 5, potted plant 2, clock 1, dog 1, dining table 1 |
| yolo26m | 8.22 | car 17, person 11, chair 8, motorcycle 5, umbrella 5, potted plant 2, dining table 2, clock 1, parking meter 1, stop sign 1, handbag 1 |

Full video with `detect_video.py` (yolo26n, 555 frames, writing the annotated video): 111.0 FPS model-only, 47.6 FPS end-to-end, 3.56 objects per frame on average, at most 9 cars and 4 people in one frame, mean confidence 0.73. Unique tracked objects: person 20, car 19, potted plant 5, motorcycle 4, chair 3, umbrella 2, clock 1, bench 1. Output: `results/annotated_street_yolo26n.mp4`.

Observations:
- **All three models run in real time** on the laptop GPU: even the largest (m) reaches ~44 FPS model-only and ~37 FPS end-to-end, above the 25–30 FPS target. Nano is ~2.8× faster than medium.
- **Bigger model → higher accuracy, mostly through recall.** mAP50-95 grows 0.484 → 0.570 → 0.617 and recall 0.55 → 0.66 → 0.68, while precision stays around 0.73–0.79. The larger models mainly find *more* objects rather than making fewer mistakes, which matches the video: 4.65 → 7.01 → 8.22 objects per frame. The extra detections are mostly classes like chair, umbrella and dining table, i.e. objects nano finds less often (typically smaller or partly hidden ones).
- **Speed trade-off is not linear.** n → s costs only ~3 ms per frame for +0.09 mAP50-95; s → m doubles the time for +0.05. For live video on this machine, **yolo26s** is the best balance; nano is the choice for weaker hardware or many parallel streams.
- **Unique counts are overestimated.** Tracking gives 20 "unique" people for the full video, and the number changes between models (11–13 people in the first 300 frames). When a person is hidden behind a car or a detection disappears for a few frames, ByteTrack may assign a new ID (ID switch), so one person is counted twice. Counts from tracking are an upper bound; reliable counting needs a stronger tracker (BoT-SORT with re-identification) or a counting line.
- **Some rare classes look like false positives** (clock, parking meter, stop sign, handbag, dog, each once). Larger models detect more classes overall, including uncertain ones; raising `--conf` or filtering classes (`--classes person car`) removes them for a real application.
- **COCO128 mAP is optimistic.** These images are from COCO train2017, which the models were trained on. Our nano result (0.484) is above the official val2017 value (0.409) for that reason; the official numbers are the fair quality reference, COCO128 only confirms the models work correctly.
- **End-to-end FPS is lower than model FPS** because of video decoding, drawing boxes and, in `detect_video.py`, encoding the output video (47.6 FPS with saving vs ~80 FPS without).
- **Parameter count:** the checkpoint (2.6M for nano) includes the training-time one-to-many head; the fused inference model used for prediction has 2.4M, as in the official table.
- **Measurement note:** on Apple GPU, PyTorch runs asynchronously, so Ultralytics' "inference" timer stops before the GPU finishes and the waiting time appears in "post-processing". Only the total per-frame time (pre + inference + post) is meaningful, which is why the table reports the total. The first validation run was also slower (12.3 vs 4.3 ms inference for nano) because of GPU kernel compilation and font-cache creation; the table uses the second run.

### 4.5 Implementation details and efficiency

- `stream=True`: frames are processed one by one as a generator → constant memory for any video length.
- Tracking with `persist=True` keeps track IDs between frames; unique objects are counted from track IDs.
- `--classes person car` filters classes inside the model (less post-processing); `--conf` sets the confidence threshold.
- The annotated video is written with OpenCV at the source FPS; `--show` displays a live window (webcam mode).
- Warm-up before timing; speeds are taken from Ultralytics' per-frame profilers.
- Limitations: only 80 COCO classes; small/far objects and heavy occlusion reduce recall; IDs can switch after long occlusions.

---

## 5. LLM — local deployment

### 5.1 Tool and its advantages

**Ollama** (built on llama.cpp).

- One command to download and run a model (`ollama pull`, `ollama run`); manages model files and versions.
- Runs **quantized GGUF** models, so an 8B model fits in ~5 GB instead of ~16 GB.
- Uses **Metal** on Apple Silicon; unified memory means the GPU can use system RAM directly.
- Local REST API (`localhost:11434`) — any language can use it; no data leaves the machine.
- Easy to swap/compare models (Llama, Qwen, Gemma, Mistral…).

This is a different framework from the text task (Transformers/PyTorch), satisfying the "at least 2 frameworks" requirement.

### 5.2 Model: how it works and why it was chosen

**Llama 3.1 8B Instruct** (Meta), 4-bit quantization (Q4_K_M) via Ollama.

How it works:
1. Text → tokens (vocabulary of 128K tokens).
2. **Decoder-only Transformer** (32 layers): each token attends only to previous tokens (causal attention). Uses Grouped-Query Attention (smaller memory for the KV-cache) and RoPE positional encoding; context up to 128K tokens (Ollama uses a smaller default context window).
3. **Autoregressive generation**: the model outputs a probability distribution for the next token, one token is chosen (greedy at temperature 0, sampled otherwise), appended, and the process repeats until an end token.
4. Pretrained on ~15T tokens of public text (next-token prediction), then instruction-tuned (supervised fine-tuning + preference optimization) to follow instructions and hold a dialog.
5. **Quantization**: weights stored in ~4 bits instead of 16 → ~4× less memory and faster on a laptop, with a small quality loss.
6. Two phases of inference: **prompt processing** (the whole prompt in parallel, fast) and **generation** (token by token, limited by memory bandwidth — that's why tokens/s is the key metric).

Why this model:
- 8B is the largest size that runs comfortably on a laptop; it's noticeably better than 1–3B models at reasoning and following instructions.
- Multilingual (handles Russian and Turkish reasonably), open weights, permissive license for research/education.
- Well supported by Ollama/llama.cpp.
- `llama3.2:3b` is used as a lighter comparison model (and as the fallback for 8 GB Macs).

### 5.3 Data

No training data is used. **Input**: a list of chat messages `{role: system|user|assistant, content}` — the full history is sent on every turn, because the model itself is stateless. **Output**: generated text + statistics (token counts, durations in ns).

Benchmark set (`benchmark.py`): 8 prompts of different types — fact, math, reasoning, code, Russian, Turkish, networking, summarization — 7 of them have an expected keyword for automatic checking.

### 5.4 Quality metrics

- **Generation speed (tokens/s)** — main infrastructure metric; ~ reading speed or faster is "usable".
- **Time to first token** / prompt processing time — perceived responsiveness.
- **Load time** — cold start.
- **Memory** (from `/api/ps`) — must fit in RAM alongside other apps.
- **Keyword pass rate** — crude correctness check (deterministic: temperature 0, fixed seed).
- **Manual review** of answers (hallucinations, language quality in RU/TR).
- For production: established benchmarks (MMLU, GSM8K) or LLM-as-a-judge.

Results:

| Model | Load, s | Prompt eval, s | Generation, tok/s (avg) | Generation on long answers, tok/s | Memory, GB | Keyword checks |
|---|---|---|---|---|---|---|
| llama3.2:3b | 1.79 | 0.08 | 80.6 | ~60 | 2.55 | 6/7 |
| llama3.1:8b | 2.58 | 0.15 | 39.0 | ~28 | 5.26 | 6/7 |

Per-prompt results:

| Prompt | llama3.2:3b | llama3.1:8b |
|---|---|---|
| fact (capital of Australia) | PASS — "Canberra." | PASS — "Canberra" |
| math (120 km / 1.5 h) | PASS — 80 | PASS — 80 |
| reasoning (3 apples − 1 + 5) | **FAIL — 8** | **FAIL — 8** |
| code (`is_prime`) | PASS | PASS |
| Russian → English translation | PASS — "I love programming." | PASS |
| Turkish question | PASS — "Ankara." | PASS — "Ankara" |
| networking (DNS transport) | PASS — "UDP." | PASS — "UDP." |
| summary | correct one-sentence summary | correct one-sentence summary |

Observations:
- **Speed vs size.** The 8B model generates about 2× slower than the 3B model and uses about 2× more memory — roughly proportional to the number of parameters.
- **Why: generation is memory-bandwidth bound.** To produce each token the model must read all its weights from memory once. M3 Pro memory bandwidth is ~150 GB/s, so the theoretical maximum is ≈ 150 / 4.9 GB ≈ 30 tok/s for the 8B model and ≈ 150 / 2.0 GB ≈ 75 tok/s for the 3B model. The measured ~28 and ~60 tok/s on long answers are close to these limits. This is also why quantization (smaller weights) directly increases speed.
- **Average tok/s is inflated by very short answers.** For one-word answers ("80", "UDP.") only a few tokens are generated, so the measurement is noisy (e.g. 110 tok/s). The longer answers (code, summary) give the realistic speed: ~60 tok/s (3B) and ~28 tok/s (8B).
- **Both speeds are comfortable for a user.** People read about 5–8 tokens per second, so even the 8B model writes several times faster than a person reads. Time to first token is under 0.2 s after loading; cold start (loading the model into memory) is ~2–2.6 s.
- **Memory.** 5.26 GB for the 8B model (weights + KV-cache for the context) leaves plenty of room on an 18 GB machine; the model runs fully on the GPU.
- **Both models made the same reasoning error.** "I have 3 apples, eat one and buy 5 more" → both answered 8 instead of 7 (they ignored "eat one" and computed 3 + 5). Because the prompt demanded "only the number", the model had no room to reason step by step; small models are especially weak at multi-step arithmetic done "in one go". This shows that LLM answers must be checked and that prompt design affects correctness.
- **Multilingual.** Both models handled the Russian translation and the Turkish question correctly.
- **Limitation of the benchmark.** The keyword test is too simple to show the quality difference between 3B and 8B (both 6/7). Larger models are generally stronger on harder reasoning and longer texts; a fair comparison would need a bigger benchmark (e.g. GSM8K, MMLU) or manual evaluation of longer answers.
- **Conclusion.** `llama3.1:8b` is chosen as the main model: on this machine it is fast enough for interactive use and, being larger, is expected to be more reliable on complex tasks. `llama3.2:3b` is a good option when speed or memory matters more (e.g. 8 GB laptops).

### 5.5 Implementation details and efficiency

- `chat.py` **streams** tokens so the answer appears immediately; keeps conversation history; `/reset` clears it.
- Checks that Ollama is running and the model is downloaded before starting.
- `benchmark.py` uses `temperature=0` and `seed=42` for reproducible results, and `num_predict=512` to cap answer length.
- The first request includes model loading — reported separately so it doesn't distort speed numbers.
- Ollama keeps the model in memory for a few minutes after a request, so later requests skip loading.
- Limitation: the context window and full history grow with every turn, so long chats get slower; could be solved with history truncation or summarization.
