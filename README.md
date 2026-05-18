# SignSpeech 🤟

### Real-Time ASL Sign Language Recognition & Voice Synthesis

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat-square)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-WLASL%20100-purple?style=flat-square)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)

A deep learning system that recognizes American Sign Language (ASL) 
from webcam video and converts recognized signs into spoken voice output 
in real time.

---

## Pipeline
Webcam Input
↓
[M2] MediaPipe Hand Detection → ROI Crop (224×224)
↓
[M3] MobileNetV2 CNN → Frame Embeddings (32 × 1280)
↓
[M4] Transformer Encoder → Predicted Class (100 words)
↓
[M5] Text-to-Speech → Spoken Output

---

## Dataset

- **Name:** WLASL (Word-Level American Sign Language)
- **Source:** [Kaggle — WLASL Processed](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)
- **Size:** ~12,000 video clips · 100 word classes (nslt_100 subset)
- **Splits:** Train / Val / Test (signer-aware — no data leakage)
- **License:** Academic and research use only · No commercial usage

---

## Results

| Model | Top-1 Acc | Top-5 Acc | Params | Latency/video |
|-------|-----------|-----------|--------|---------------|
| CNN + MLP (baseline) | — | — | — | 13.11 ms |
| CNN + BiLSTM | — | — | 4.78M | — |
| CNN + Transformer | — | — | 15.88M | — |

*Results will be updated after training completes.*

---

## Team

| Member | Role | Notebooks |
|--------|------|-----------|
| M1 | Data pipeline & VideoDataLoader | `M1_WLASL_DataPipeline.ipynb` |
| M2 | Hand detection & ROI cropping | `M2_handDetect_fixed.ipynb` |
| M3 | CNN frame encoder (MobileNetV2) | `M3_SpatialFeatureExtractor.ipynb` |
| M4 | Transformer + BiLSTM + training loop | `M4_TemporalModels.ipynb` |
| M5 | Demo app, TTS integration & report | `M5_Demo.ipynb` |

---

---

## Quick Start

```bash
git clone https://github.com/yourteam/SignSpeech.git
cd SignSpeech
pip install -r requirements.txt
```

> **Dataset setup:** Download WLASL from 
> [Kaggle](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed) 
> and place the `videos/` folder and JSON files in the project root.
> The raw videos are not included in this repository.

---

## Key Design Decisions

- **T = 32 frames** per clip — uniform sampling from the sign window
- **num_workers = 0** — required to prevent MediaPipe/PyTorch multiprocessing deadlock
- **Signer-aware splits** — the same signer never appears in both train and test
- **Frozen MobileNetV2** backbone — fine-tuning available via `freeze_backbone=False`
- **100-class subset** — using `nslt_100.json` for manageable training time

## Data

### Source & Collection

The dataset used is **WLASL (Word-Level American Sign Language)**, the largest 
publicly available video dataset for word-level ASL recognition. It was collected 
from YouTube and ASL educational platforms, featuring multiple signers per word 
to ensure diversity across age, gender, and signing style.

- **Full dataset:** 21,083 video clips · 2,000 word classes
- **Subset used:** `nslt_100.json` — 100 most frequent words · ~1,013 usable clips
- **Download:** [Kaggle — WLASL Processed](https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed)
- **Format:** `.mp4` videos · all videos in a single flat `videos/` folder
- **Metadata:** `WLASL_v0.3.json` — word labels, frame windows, signer IDs per clip

We did not collect or record any data ourselves. All videos are pre-existing 
recordings of ASL signers. A `missing.txt` file lists video IDs that were 
unavailable at download time — these are silently skipped during loading.

---

### Dataset Split

Splits are taken directly from the WLASL JSON metadata and are **signer-aware** — 
the same signer never appears in both train and test sets. This prevents the model 
from memorizing individual signing styles rather than the signs themselves.

| Split | Clips | Batches (batch size 8) |
|-------|-------|------------------------|
| Train | 748 | 93 |
| Val | 165 | 21 |
| Test | 100 | 13 |
| **Total** | **1,013** | — |

> ⚠️ Do not re-split the data randomly. The signer-aware split is intentional 
> and critical for valid evaluation.

---

### Preprocessing Pipeline

Each raw video goes through the following steps before reaching the model:

**Step 1 — Frame extraction (M1)**  
The sign segment is identified using `frame_start` and `frame_end` from the 
JSON metadata. Exactly **32 frames** are uniformly sampled from this window 
regardless of the original video length. Videos shorter than 32 frames are 
handled gracefully — uniform sampling naturally repeats nearby frames.

**Step 2 — Hand detection & ROI crop (M2)**  
Each frame is passed through **MediaPipe HandLandmarker** (Tasks API). 
The 21 detected hand landmarks define a bounding box around the hand region. 
A **20% padding** is added on each side to include wrist context. The crop 
is resized to **224 × 224 pixels**.

Fallback logic when detection fails:
- If a frame fails detection → reuse the previous frame's hand crop
- If the very first frame fails → use the full resized frame as-is

**Step 3 — Normalization**  
Pixel values are normalized from `[0, 255]` to `[0.0, 1.0]`. Color channels 
are converted from BGR (OpenCV default) to RGB before passing to MediaPipe 
and MobileNetV2.

**Final tensor shape per clip:**
```
(32, 3, 224, 224)  →  T frames · RGB channels · 224×224 pixels
```

**Final batch shape:**
```
frames : torch.FloatTensor  (8, 32, 3, 224, 224)  values in [0.0, 1.0]
labels : torch.LongTensor   (8,)                   integer class index [0–99]
```

---

### Sample Data

The 100-word vocabulary covers common everyday ASL signs including:

```
accident · book · candy · chair · clothes · color · computer · cook
corn · day · deer · drink · eat · family · fish · fruit · go · happy
help · hot · no · nurse · orange · play · school · study · who · woman · year
```

Example clips from a single batch (verified output):
```
['who', 'study', 'play', 'school', 'year', 'accident', 'corn', 'woman']
```

---

### Data Constraints & Limitations

| Constraint | Detail |
|---|---|
| Class imbalance | Some words have significantly more clips than others — the distribution is long-tailed |
| Missing videos | A portion of the original WLASL clips were unavailable at download time and are excluded |
| Detection failures | MediaPipe fails on ~5–15% of frames depending on lighting — fallback logic handles these |
| Single flat folder | All ~12,000 videos share one `videos/` directory — no subfolders by class |
| Academic license | WLASL data is for research only · commercial use is prohibited |
| `num_workers = 0` | MediaPipe's internal threading conflicts with PyTorch multiprocessing — parallel loading is disabled |
---

## Course

Artificial Neural Networks — Nile University  
Project: SignSpeech ASL Recognition & Voice Synthesis
