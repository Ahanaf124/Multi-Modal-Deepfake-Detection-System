# Multi-Modal Deepfake Detection System

A late-fusion deepfake detection system that classifies media as real or fake across **three modalities** — image, audio, and text — by combining three independently trained deep learning models.

> Final-year honours dissertation project — BSc (Hons) Computer Science, London South Bank University (CSI-6-CSP, 2024/25).
> **Author:** Ahanaf Alam

---

## Overview

| Modality | Backbone | Task |
|---|---|---|
| Image | EfficientNet-B0 (ImageNet pretrained) | Detect facial manipulation in 224×224 RGB images |
| Audio | ResNet18 adapted to 1-channel input | Classify Mel-spectrograms of 3-second audio clips |
| Text | DeBERTa-v3-small | Classify human- vs AI-generated text |
| Fusion | Performance-weighted softmax ensemble | Combine per-modality scores into a final decision |

Each unimodal detector is trained and evaluated independently, then late-fused at inference time. The architecture is modular — components can be replaced or upgraded without retraining the others.

### Reported test results

| Model | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Image (EfficientNet-B0) | 98.86% | 0.9885 | 0.9993 |
| Audio (ResNet18) | 85.71% | — | 1.000 |
| Text (DeBERTa-v3-small) | trained on 29,145 samples | — | — |

> Audio metrics are computed on a held-out test set of n=14. See `bootstrap_audio.py` for confidence intervals.

---

## Repository contents

```
.
├── app.py                      Streamlit demo UI for inference
├── run_training.py             Standalone training script (all three models)
├── deepfake_detection.ipynb    Full pipeline notebook (env setup → export)
├── bootstrap_audio.py          Bootstrap CIs for the small audio test set
├── requirements.txt            Python dependencies
├── training_log.txt            Recorded training output
├── structure.md                Dissertation chapter/section plan
└── README.md                   This file
```

The trained model weights (`.pth`), datasets, and exported artefacts are **not** committed (see `.gitignore`). They are produced by running the notebook or `run_training.py`.

---

## Datasets

All datasets are sourced from Kaggle (publicly available, academic use):

| Modality | Kaggle path |
|---|---|
| Image | `manjilkarki/deepfake-and-real-images` |
| Audio | `birdy654/deep-voice-deepfake-voice-recognition` |
| Text  | `sunilthite/llm-detect-ai-generated-text-dataset` |

Download via the Kaggle CLI into `./data/video`, `./data/audio`, and `./data/text` respectively (the notebook does this automatically given a `kaggle.json`).

---

## Quick start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd <repo>
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for training. The Streamlit app will run on CPU.

### 2. Train (optional — only if you want to reproduce the weights)

```bash
python run_training.py
```

Or step through `deepfake_detection.ipynb` cell by cell. Trained weights are written to `./models/` and exported to `./exported_models/`.

### 3. Run the demo app

```bash
streamlit run app.py
```

The app loads weights from `./exported_models/` (with a fallback to `./models/`) and provides browser tabs for image, audio, and text inference.

---

## Project structure (runtime)

```
data/                Kaggle downloads (gitignored)
  ├── video/
  ├── audio/
  └── text/
models/              Per-modality .pth files saved during training (gitignored)
exported_models/     Final exports + metadata.json (gitignored)
```

---

## Method summary

- **Splits:** 70 / 10 / 20 train/val/test, stratified, fixed seed 42.
- **Image preprocessing:** resize 224×224, ImageNet normalisation; train-time random horizontal flip, ±10° rotation, colour jitter.
- **Audio preprocessing:** load at 16 kHz, fixed 3-second window (pad/trim), 128-bin Mel-spectrogram, power-to-dB.
- **Text preprocessing:** DeBERTa-v3 tokenizer, max length 256.
- **Fusion:** each model emits a softmax `P(fake)`; the fusion engine produces a performance-weighted average and applies a 0.5 threshold.

See `structure.md` and the notebook for the full design rationale, evaluation methodology, and per-chapter dissertation plan.

---

## Limitations

- The audio dataset is small (66 files total, 14 in the test set) and class-imbalanced — bootstrap CIs are wide. See `bootstrap_audio.py`.
- Models are evaluated on the same Kaggle distributions used for training; out-of-distribution generalisation has not been validated.
- The system handles pre-recorded media only (no live-stream / real-time inference).
- No explainability layer (e.g. GradCAM, attention visualisation) is included.

---

## Acknowledgements

LSBU School of Engineering — supervisor and second assessor.
Kaggle dataset authors as listed above.
PyTorch, HuggingFace Transformers, librosa, facenet-pytorch.

---

*For academic context, dissertation chapter plan, and the full evaluation write-up see `structure.md`.*
