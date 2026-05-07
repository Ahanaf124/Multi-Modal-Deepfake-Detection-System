#!/usr/bin/env python
"""
Multi-Modal Deepfake Detection — Streamlit demo app.

Loads the three trained models from ./exported_models/ and exposes a
browser UI that runs inference on user-supplied text, images, and audio.

Author: Ahanaf Alam
"""

from __future__ import annotations

import io
import json
import warnings
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# Paths and device
# ------------------------------------------------------------------ #
ROOT = Path(__file__).resolve().parent
EXPORT_DIR = ROOT / "exported_models"
FALLBACK_DIR = ROOT / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directories to skip while scanning (env / cache / data dumps)
_SKIP_DIRS = {".venv", "venv", "env", ".git", "__pycache__", "node_modules", "data", ".cache"}


def _find_weights(*candidate_names: str) -> Path | None:
    """Locate a weights file. Checks conventional locations first, then
    falls back to a recursive scan of the project tree so users can drop
    the .pth files anywhere under the project folder."""
    # 1. Conventional locations
    for name in candidate_names:
        for base in (EXPORT_DIR, FALLBACK_DIR):
            p = base / name
            if p.exists():
                return p
    # 2. Recursive scan as a fallback
    wanted = set(candidate_names)
    for p in ROOT.rglob("*.pth"):
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.name in wanted:
            return p
    return None


def _find_file(name: str) -> Path | None:
    """Locate any file by exact name, scanning the project tree."""
    p = EXPORT_DIR / name
    if p.exists():
        return p
    for found in ROOT.rglob(name):
        if any(part in _SKIP_DIRS for part in found.parts):
            continue
        if found.is_file():
            return found
    return None


IMAGE_WEIGHTS = _find_weights("image_model.pth", "best_image_model.pth")
AUDIO_WEIGHTS = _find_weights("audio_model.pth", "best_audio_model.pth")
TEXT_WEIGHTS = _find_weights("text_model.pth", "best_text_model.pth")
META_PATH = _find_file("metadata.json")


# ------------------------------------------------------------------ #
# Model architectures (must match run_training.py)
# ------------------------------------------------------------------ #
class ImageDetector(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        from torchvision import models as tvm

        self.backbone = tvm.efficientnet_b0(weights=None)
        in_feat = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_feat, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class AudioDetector(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        from torchvision import models as tvm

        self.backbone = tvm.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_feat = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_feat, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


class TextDetector(nn.Module):
    def __init__(self, model_name: str = "microsoft/deberta-v3-small", num_classes: int = 2):
        super().__init__()
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


# ------------------------------------------------------------------ #
# Cached loaders
# ------------------------------------------------------------------ #
def _load_state(path: Path) -> dict:
    """Load a checkpoint and unwrap common training-script dict layouts."""
    obj = torch.load(path, map_location=DEVICE, weights_only=False)
    if isinstance(obj, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            inner = obj.get(key)
            if isinstance(inner, dict):
                return inner
    return obj


@st.cache_resource(show_spinner="Loading image model…")
def load_image_model():
    if IMAGE_WEIGHTS is None:
        return None
    m = ImageDetector().to(DEVICE)
    m.load_state_dict(_load_state(IMAGE_WEIGHTS))
    m.eval()
    return m


@st.cache_resource(show_spinner="Loading audio model…")
def load_audio_model():
    if AUDIO_WEIGHTS is None:
        return None
    m = AudioDetector().to(DEVICE)
    m.load_state_dict(_load_state(AUDIO_WEIGHTS))
    m.eval()
    return m


@st.cache_resource(show_spinner="Loading text model + tokenizer (first run downloads ~280 MB)…")
def load_text_model():
    if TEXT_WEIGHTS is None:
        return None, None
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("microsoft/deberta-v3-small")
    m = TextDetector().to(DEVICE)
    m.float()  # transformers 5.x may download HF backbones in fp16; force fp32
    m.load_state_dict(_load_state(TEXT_WEIGHTS))
    m.eval()
    return m, tok


@st.cache_data
def load_metadata():
    if META_PATH is not None and META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text())
        except Exception:
            return None
    return None


def _missing_weights_error(kind: str, names: tuple[str, ...]) -> None:
    name_list = " or ".join(f"`{n}`" for n in names)
    st.error(
        f"**{kind} weights not found.** Looked for {name_list} under "
        f"`{ROOT}` (scanned recursively).\n\n"
        f"Drop the `.pth` file anywhere inside `{ROOT}` (e.g. into "
        f"`exported_models/`) and refresh the page."
    )


# ------------------------------------------------------------------ #
# Inference helpers
# ------------------------------------------------------------------ #
def _verdict(p_fake: float) -> tuple[str, str]:
    if p_fake >= 0.5:
        return "FAKE / AI-GENERATED", "🔴"
    return "REAL / HUMAN", "🟢"


def predict_image(file_bytes: bytes):
    from PIL import Image
    from torchvision import transforms

    model = load_image_model()
    if model is None:
        _missing_weights_error("Image", ("image_model.pth", "best_image_model.pth"))
        return None

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    x = tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).cpu().numpy()[0]
    return img, probs


def predict_audio(file_bytes: bytes, filename: str):
    import librosa
    import soundfile as sf

    model = load_audio_model()
    if model is None:
        _missing_weights_error("Audio", ("audio_model.pth", "best_audio_model.pth"))
        return None

    sr_target = 16000
    duration = 3.0
    max_len = int(sr_target * duration)

    try:
        with io.BytesIO(file_bytes) as buf:
            y, sr = sf.read(buf, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != sr_target:
            y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
    except Exception:
        # mp3 / m4a fall through to librosa+audioread
        tmp = ROOT / "_tmp_upload"
        tmp.mkdir(exist_ok=True)
        path = tmp / filename
        path.write_bytes(file_bytes)
        y, _ = librosa.load(str(path), sr=sr_target, duration=duration)

    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]

    mel = librosa.feature.melspectrogram(y=y, sr=sr_target, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1).cpu().numpy()[0]
    return y, sr_target, mel_db, probs


def predict_text(text: str):
    model, tok = load_text_model()
    if model is None:
        _missing_weights_error("Text", ("text_model.pth", "best_text_model.pth"))
        return None

    enc = tok(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    ids = enc["input_ids"].to(DEVICE)
    mask = enc["attention_mask"].to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(ids, mask), dim=1).cpu().numpy()[0]
    return probs


# ------------------------------------------------------------------ #
# UI
# ------------------------------------------------------------------ #
st.set_page_config(page_title="Multi-Modal Deepfake Detector", page_icon="🕵️", layout="wide")
st.title("🕵️ Multi-Modal Deepfake Detection System")
st.caption(
    f"Image · Audio · Text — running on **{DEVICE.type.upper()}** "
    f"({torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'CPU'})"
)

meta = load_metadata()
if meta:
    cols = st.columns(3)
    for col, key in zip(cols, ("image", "audio", "text")):
        if key in meta.get("models", {}):
            m = meta["models"][key]
            col.metric(
                label=f"{key.capitalize()} ({m['architecture']})",
                value=f"{m['accuracy']:.2f}%",
                help=f"F1 {m['f1']:.4f} · AUC {m['auc']:.4f}",
            )

with st.expander("📂 Weights resolved at startup", expanded=False):
    for label, path in (
        ("Text",  TEXT_WEIGHTS),
        ("Image", IMAGE_WEIGHTS),
        ("Audio", AUDIO_WEIGHTS),
        ("Metadata", META_PATH),
    ):
        st.write(f"- **{label}:** `{path}`" if path else f"- **{label}:** _not found_")

tab_text, tab_image, tab_audio = st.tabs(["📝 Text", "🖼️ Image", "🎵 Audio"])

# --- Text tab ---
with tab_text:
    st.subheader("Detect AI-generated text")
    sample = st.text_area(
        "Paste text to analyse",
        height=200,
        placeholder="Paste an essay, article, or paragraph here…",
    )
    if st.button("Analyse text", type="primary", disabled=not sample.strip()):
        out = predict_text(sample)
        if out is not None:
            p_real, p_fake = float(out[0]), float(out[1])
            label, icon = _verdict(p_fake)
            st.markdown(f"### {icon} Verdict: **{label}**")
            c1, c2 = st.columns(2)
            c1.metric("P(human)", f"{p_real*100:.2f}%")
            c2.metric("P(AI-generated)", f"{p_fake*100:.2f}%")
            st.progress(p_fake)

# --- Image tab ---
with tab_image:
    st.subheader("Detect deepfake images")
    img_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png", "bmp"])
    if img_file is not None and st.button("Analyse image", type="primary"):
        out = predict_image(img_file.getvalue())
        if out is not None:
            img, probs = out
            p_real, p_fake = float(probs[0]), float(probs[1])
            label, icon = _verdict(p_fake)
            c1, c2 = st.columns([1, 1])
            c1.image(img, caption=img_file.name, use_container_width=True)
            with c2:
                st.markdown(f"### {icon} Verdict: **{label}**")
                st.metric("P(real)", f"{p_real*100:.2f}%")
                st.metric("P(fake)", f"{p_fake*100:.2f}%")
                st.progress(p_fake)

# --- Audio tab ---
with tab_audio:
    st.subheader("Detect deepfake / synthetic audio")
    aud_file = st.file_uploader(
        "Upload a voice clip (≥3 s recommended)", type=["wav", "mp3", "flac", "ogg", "m4a"]
    )
    if aud_file is not None and st.button("Analyse audio", type="primary"):
        out = predict_audio(aud_file.getvalue(), aud_file.name)
        if out is not None:
            y, sr, mel_db, probs = out
            p_real, p_fake = float(probs[0]), float(probs[1])
            label, icon = _verdict(p_fake)

            st.audio(aud_file.getvalue())
            st.markdown(f"### {icon} Verdict: **{label}**")
            c1, c2 = st.columns(2)
            c1.metric("P(real / human)", f"{p_real*100:.2f}%")
            c2.metric("P(fake / synthetic)", f"{p_fake*100:.2f}%")
            st.progress(p_fake)

            try:
                import matplotlib.pyplot as plt
                import librosa.display

                fig, ax = plt.subplots(figsize=(8, 3))
                librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax)
                ax.set_title("Mel-Spectrogram (model input)")
                st.pyplot(fig, clear_figure=True)
            except Exception:
                pass

st.markdown("---")
with st.expander("ℹ️  Model details"):
    st.write(
        f"""
        - **Image:** EfficientNet-B0 fine-tuned on Kaggle *deepfake-and-real-images*.
        - **Audio:** ResNet18 over 128-bin Mel-spectrograms, Kaggle *Deep Voice* dataset (n=14 test → bootstrap CI; see `bootstrap_audio.py`).
        - **Text:** DeBERTa-v3-small fine-tuned on Kaggle *LLM-Detect-AI-Generated-Text*.
        - Weights resolved from: `{EXPORT_DIR}` (fallback `{FALLBACK_DIR}`).
        """
    )
