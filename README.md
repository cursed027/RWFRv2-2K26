---

# NTIRE 2026 – Identity-Aware Blind Face Restoration Pipeline

A modular, multi-stage blind face restoration framework designed for **robust real-world restoration** while preserving **facial identity consistency**.

This pipeline combines identity verification, degradation routing, diffusion restoration, and perceptual quality selection into a unified NTIRE-style inference system.

---

## ⭐ Key Features

* Identity-aware restoration routing
* Multi-model adaptive execution
* Diffusion + GAN hybrid restoration
* Automatic quality-based output selection
* Fully inference-based (no training required)
* Modular NTIRE-style pipeline

---

## 📌 Pipeline Architecture

```
Input LQ Image
        │
        ▼
Stage-0 : Face Detection + Alignment
        │
        ▼
Stage-1 : GFPGAN Identity Anchor Generation
        │
        ▼
Stage-2 : Identity + Quality Measurement
        │
        ▼
Stage-3 : Thresholding
        │
        ▼
Stage-4 : Routing Decision
        │
        ▼
Stage-5 : Model Execution (DiffBIR / CodeFormer)
        │
        ▼
Stage-6 : NR-IQA + Identity Selection
        │
        ▼
Final Output
```

---

## 🧩 Stage Descriptions

| Stage   | Purpose                                                                 |
| ------- | ----------------------------------------------------------------------- |
| Stage-0 | Detects faces and generates aligned crops using InsightFace             |
| Stage-1 | Generates identity-preserving anchor restoration using GFPGAN           |
| Stage-2 | Computes identity similarity and quality metrics                        |
| Stage-3 | Applies adaptive thresholds for routing                                 |
| Stage-4 | Determines restoration strategy based on degradation and identity state |
| Stage-5 | Executes DiffBIR and/or CodeFormer based on routing                     |
| Stage-6 | Selects best output using NR-IQA + identity consistency                 |

---

## 🧠 Routing Logic Summary

Pipeline dynamically chooses restoration strategy based on:

* Identity preservation confidence
* Estimated degradation severity

### Example Routing:

| Case | Identity   | Degradation   | Action               |
| ---- | ---------- | ------------- | -------------------- |
| A    | Pass       | Mild/Moderate | DiffBIR → CodeFormer |
| B    | Pass       | Severe        | DiffBIR only         |
| C    | Fail       | Mild          | Use GFPGAN Anchor    |
| D    | Unknown    | Any           | Light DiffBIR        |
| E    | Worst Case | Any           | Anchor fallback      |

---

## 📂 Repository Structure

```
repo/
├── adaface/                 # Identity embedding model
├── face_alignment/          # Face landmark utilities
├── models/                  # Model wrappers
├── third_party/
│   ├── GFPGAN/
│   ├── CodeFormer/
│   └── DiffBIR/
├── stage0.py
├── stage2.py
├── stage3.py
├── stage4.py
├── stage5.py
├── stage6.py
├── run.py                   # Main pipeline runner
├── req.txt
└── .gitignore
```

---

## 🖥️ Environment Requirements

### Python

```
Python 3.10 ONLY
```

Python 3.11+ will break dependencies.

---

### Hardware

| Component        | Requirement         |
| ---------------- | ------------------- |
| GPU              | CUDA capable        |
| Recommended VRAM | ≥16GB (for DiffBIR) |
| OS               | Linux Recommended   |

---

## ⚙️ Installation Guide

---

### 1️⃣ Install PyTorch FIRST

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
--extra-index-url https://download.pytorch.org/whl/cu118
```

Verify:

```bash
python - << EOF
import torch
print(torch.cuda.is_available())
EOF
```

---

### 2️⃣ Install Dependencies

```
pip install -r req.txt
```

---

### 3️⃣ Install GFPGAN (Mandatory)

```
pip install -e third_party/GFPGAN
```

This registers BasicSR modules correctly.

---

### 4️⃣ Environment Verification

```
python - << EOF
from gfpgan import GFPGANer
from basicsr.archs.stylegan2_arch import ResBlock
print("Environment OK")
EOF
```

---

## 🚀 Running The Pipeline

---

### Routing Only (Fast Debug Mode)

```
python run.py \
--input /path/to/images \
--workdir work
```

---

### Full Restoration Pipeline

```
python run.py \
--input /path/to/images \
--workdir work \
--execute \
--final_out work/final_outputs
```

---

### Full Pipeline + Output Selection

```
python run.py \
--input /path/to/images \
--workdir work \
--execute \
--final_out work/final_outputs \
--select
```

---

## 📊 Stage-6 Selection Metrics

Final output is selected using weighted combination of:

* AdaFace identity similarity
* NIQE
* CLIP-IQA
* MANIQA
* MUSIQ

Pipeline chooses restoration output that best balances:

```
Identity Preservation + Perceptual Quality
```

---

## 📥 Pretrained Weight Handling

Weights are **automatically downloaded** during first inference run.

Models:

* InsightFace
* GFPGAN
* CodeFormer
* DiffBIR

Manual download is required ony for 
* AdaFace (adaface_ir50_ms1mv2.ckpt)
  ```
  !gdown --id 1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI
  ```
  - store this weight at pretrained/
---

## ⚠️ Important Notes

* DiffBIR is GPU memory intensive
* TorchVision warnings can be ignored
* Temporary `_tmp_*` folders are internal execution artifacts
* Final outputs are stored in:

```
final_outputs/
```

---

## 🧪 Design Philosophy

This pipeline treats blind face restoration as:

> Decision + Restoration + Verification

Rather than applying a single restoration model blindly.

---

## 🔬 NTIRE Motivation

Real-world degradation varies significantly.
Identity-aware routing improves robustness across:

* Motion blur
* Low-light noise
* Compression artifacts
* Mixed degradations

---

## 📜 License

Free To Use

---

## 🙌 Acknowledgements

* InsightFace
* GFPGAN
* CodeFormer
* DiffBIR
* AdaFace
* PyIQA

---

# ⭐ Suggested Citation (Future)

```
Coming soon
```

---

# 💡 Research Status

Current repository supports:

* Full inference pipeline
* Adaptive routing
* Identity-aware selection

Training extensions are under development.

---

# 📬 Contact

Author: Milan Kumar Singh

---


