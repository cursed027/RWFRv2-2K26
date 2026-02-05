# stage2.py
import os
import cv2
import csv
import argparse
import numpy as np
from typing import List, Dict

IMG_EXTS = (".jpg", ".png", ".jpeg")


# ============================================================
# Utility
# ============================================================

def list_images(root):
    return sorted([
        f for f in os.listdir(root)
        if f.lower().endswith(IMG_EXTS)
    ])


def load_img(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def find_aligned_anchor(name, aligned_faces_dir):
    """
    GFPGAN aligned outputs often append suffixes like _00.
    Match by filename prefix.
    """
    stem = os.path.splitext(name)[0]
    if not os.path.isdir(aligned_faces_dir):
        return None

    for f in os.listdir(aligned_faces_dir):
        if f.startswith(stem) and f.lower().endswith(IMG_EXTS):
            return os.path.join(aligned_faces_dir, f)
    return None

# ============================================================
# Degradation measurement (identity-blind)
# ============================================================

def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def noise_score(img):
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    hf = img.astype(np.float32) - blur.astype(np.float32)
    return np.mean(np.abs(hf))


def compute_degradation_scalar(img):
    """
    Conservative, fixed normalization.
    Reviewer-safe, dataset-agnostic.
    """
    B = blur_score(img)
    N = noise_score(img)

    Bn = np.clip(B / 300.0, 0.0, 1.0)
    Nn = np.clip(N / 20.0, 0.0, 1.0)

    # Blur is less dangerous than noise for identity
    S = 0.4 * (1.0 - Bn) + 0.6 * Nn
    return float(S)


def bucket_degradation(S):
    if S < 0.35:
        return "mild"
    elif S < 0.65:
        return "moderate"
    else:
        return "severe"


# ============================================================
# AdaFace identity
# ============================================================

def load_adaface():
    """
    Load AdaFace ONCE inside Stage-2.
    Later reused by Stage-5 via refactor.
    """
    from models.adaface import AdaFaceModel
    return AdaFaceModel()



def compute_identity(aligned_lq, aligned_anchor, adaface_model):
    """
    Identity is ONLY measured on aligned geometry.
    """
    emb1 = adaface_model(aligned_lq)
    emb2 = adaface_model(aligned_anchor)

    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)

    return float(np.dot(emb1, emb2))


# ============================================================
# Stage-2 core
# ============================================================
def list_all_candidates(stage0_dir, raw_faces_dir):
    names = set()
    if os.path.isdir(stage0_dir):
        names.update(list_images(stage0_dir))
    if os.path.isdir(raw_faces_dir):
        names.update(list_images(raw_faces_dir))
    return sorted(names)

def run_stage2(
    stage0_dir: str,
    gfpgan_aligned_dir: str,
    gfpgan_raw_dir: str,
    adaface_model
) -> List[Dict]:

    aligned_faces_dir = os.path.join(gfpgan_aligned_dir, "restored_faces")
    raw_faces_dir = os.path.join(gfpgan_raw_dir, "restored_imgs")

    results = []

    for name in list_all_candidates(stage0_dir, raw_faces_dir):
        aligned_lq_path = os.path.join(stage0_dir, name)
        if os.path.exists(aligned_lq_path):
            aligned_lq = load_img(aligned_lq_path)
        else:
            aligned_lq = None

        aligned_anchor_path = find_aligned_anchor(name, aligned_faces_dir)
        raw_anchor_path = find_aligned_anchor(name, raw_faces_dir)

        # ---------------- Anchor resolution ----------------

        if aligned_anchor_path is not None and aligned_lq is not None:
            anchor_img = load_img(aligned_anchor_path)
            anchor_type = "aligned"
            I_base = compute_identity(aligned_lq, anchor_img, adaface_model)

        elif os.path.exists(raw_anchor_path):
            anchor_img = load_img(raw_anchor_path)
            anchor_type = "raw"
            I_base = None  # identity unavailable

        else:
            anchor_type = "missing"
            I_base = None


        # ---------------- Degradation -------------
        if aligned_lq is not None:
            S = compute_degradation_scalar(aligned_lq)
            D = bucket_degradation(S)
        else:
            # No aligned geometry → maximum risk
            S = 1.0
            D = "severe"


        results.append({
            "image": name,
            "anchor_type": anchor_type,
            "I_base": I_base,
            "S": S,
            "D": D
        })
        
    assert len(results) > 0, "Stage-2 produced zero samples"
    has_raw = any(r["anchor_type"] == "raw" for r in results)
    has_aligned = any(r["anchor_type"] == "aligned" for r in results)

    print(f"[Stage-2] anchors: aligned={has_aligned}, raw={has_raw}")



    return results


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Stage-2 Measurement (Identity + Degradation)")
    parser.add_argument("--stage0", required=True, help="stage0_aligned directory")
    parser.add_argument("--gfpgan_aligned", required=True, help="stage1_gfpgan_aligned directory")
    parser.add_argument("--gfpgan_raw", required=True, help="stage1_gfpgan_raw directory")
    parser.add_argument("--out", required=True, help="output CSV path")
    args = parser.parse_args()

    print("[Stage-2] Loading AdaFace...")
    adaface_model = load_adaface()

    print("[Stage-2] Running measurement...")
    results = run_stage2(
        stage0_dir=args.stage0,
        gfpgan_aligned_dir=args.gfpgan_aligned,
        gfpgan_raw_dir=args.gfpgan_raw,
        adaface_model=adaface_model
    )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "anchor_type", "I_base", "S", "D"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"[Stage-2] Done. {len(results)} samples written to {args.out}")
