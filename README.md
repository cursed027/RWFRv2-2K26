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

### Installing CodeFormer

```
cd third_party/CodeFormer
python basicsr/setup.py develop
cd ../../
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
 
  -  change path in models/adaface.py :
                    ```
             # Load pretrained weights (exactly like inference.py)
                ckpt = torch.load(
                    "/media/admin1/DL/MILAN/3StageRWFR/repo/pretrained/adaface_ir50_ms1mv2.ckpt",
                    map_location=device
                )["state_dict"]
            ```
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

# RWFR NTIRE Pipeline — Project Failure Postmortem

## 🎯 Objective

Build a **Top-3 competitive pipeline** for the NTIRE RWFR challenge using a **3-stage architecture**:

1. **Stage-1 (Identity / Structure)**
   StyleGAN2-ADA + pSp inversion
2. **Stage-2 (Texture / Realism)**
   DiffBIR (diffusion-based restoration)
3. **Stage-3 (Naturalness / Harmonization)**
   Low-strength diffusion refinement
4. *(Planned)* Stage-4
   Metric-aligned naturalness correction (noise prior)

---

## 🧠 Final Pipeline Design

```text
Input Image
   ↓
[Alignment (optional, gated)]
   ↓
IF alignment valid:
    → Stage-1 (pSp)
ELSE:
    → Skip Stage-1
   ↓
Stage-2 (DiffBIR)
   ↓
Stage-3 (Low-strength refinement)
   ↓
Stage-4 (Noise harmonization - planned)
   ↓
Final Output
```

---

## ⚠️ Key Challenges Faced

### 1. Face Alignment Instability

We attempted multiple alignment strategies using dlib 5-point landmarks:

* Eye-angle based rotation (atan2)
* Eye vs mouth vertical ordering
* Nose vs eyes invariant
* Combined heuristics

#### Outcome:

* Inconsistent results (≈50% incorrect rotations)
* Flip-flop behavior due to conflicting logic
* Sensitivity to degraded inputs (blur, children, profile)

---

### 2. Landmark Reliability Breakdown

Root issue discovered:

> dlib 5-point landmarks are **not reliable for orientation** on RWFR data

Problems observed:

* Eye order ambiguity
* Nose drift in low-quality images
* Mouth misplacement in children faces
* Detector inconsistency across datasets (LFW, Wider, CelebA, etc.)

#### Conclusion:

No geometric heuristic can fix incorrect landmarks.

---

### 3. Over-Engineering Alignment

Major mistake:

> Treating alignment as a problem to “perfect”

Symptoms:

* Multiple competing orientation rules
* Repeated redesign of logic
* Increasing complexity without improving robustness

#### Result:

* System instability increased
* Debugging complexity exploded
* Time wasted on diminishing returns

---

### 4. Logical Conflict (Critical Bug)

At one point, pipeline had:

* Nose-based rotation
* AND angle-based rotation

This caused:

* Double rotations
* Cancellation effects
* 50/50 output inconsistency

---

### 5. Misguided Fix Attempts

Rejected approaches:

* ❌ “Rotate everything 180° at end”
  → Guarantees ~50% failure

* ❌ More heuristic stacking
  → Amplifies noise in bad landmarks

* ❌ Perfect alignment goal
  → Not achievable with given tools

---

## ✅ Key Realizations (Turning Points)

### 🔑 1. Alignment is not mandatory

> Alignment is an **optimization**, not a requirement

* pSp benefits from alignment
* DiffBIR does NOT require alignment

---

### 🔑 2. Landmark failure is unavoidable

> Bad landmarks cannot be fixed downstream

Therefore:

* Stop trying to correct them
* Detect and bypass instead

---

### 🔑 3. Robust systems use fallback paths

> Strong pipelines handle failure, not eliminate it

---

## 🛠️ Final Solution Strategy

### ✅ Simplified Alignment

* Only perform:

  * Face-centered crop
  * Optional 180° correction (basic check)
* No angle-based rotation
* No overfitting heuristics

---

### ✅ Add Fallback Mechanism

```python
if alignment_valid:
    output = Stage1_pSp(image)
else:
    output = image  # bypass Stage-1

output = Stage2_DiffBIR(output)
output = Stage3_refinement(output)
```

---

### ✅ Define “Alignment Validity” (Practical Checks)

Use simple rules:

* Face detected
* Landmarks present
* Face size above threshold
* No extreme distortion

No complex geometry checks.

---

## 📊 Observed Metrics (Alignment Stage)

* Total test images: 100
* Successfully aligned: ~94
* Skipped (no face / bad landmarks): ~6

### Interpretation:

* ~6% failure rate is **acceptable**
* NTIRE pipelines tolerate such cases via fallback

---

## ❌ Mistakes Summary

| Category    | Mistake                                        |
| ----------- | ---------------------------------------------- |
| Design      | Treated alignment as mandatory                 |
| Logic       | Used multiple conflicting rotation rules       |
| Assumption  | Trusted landmark accuracy blindly              |
| Strategy    | Tried to “fix” bad data instead of bypassing   |
| Engineering | Over-complicated simple stage                  |
| Time        | Spent too long optimizing low-impact component |

---

## ✅ Correct Engineering Principles Learned

* Use **minimal necessary transformations**
* Avoid stacking heuristics on noisy signals
* Prefer **fallback over forced correction**
* Separate:

  * *core pipeline*
  * *failure handling*
* Optimize where it matters (Stages 2 & 3)

---

## 🚀 Current Status

### ✔ Completed

* Alignment pipeline (stable enough)
* Failure understanding
* Strategy correction

### 🔄 Next Steps

1. Implement **pSp inference (Stage-1)**
2. Integrate fallback routing
3. Validate identity preservation
4. Move to DiffBIR integration

---

## 🧾 Final Verdict

This phase revealed a critical insight:

> The bottleneck is not model performance — it is **pipeline robustness**.

By shifting from:

* “fix everything” → to → “handle failures gracefully”

the system becomes:

* stable
* scalable
* competition-ready

---

## 🧠 One-line takeaway

> **Do not fight bad data — route around it.**

# 📬 Contact

Author: Milan Kumar Singh

---
