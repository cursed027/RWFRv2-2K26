import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import csv
import argparse
from PIL import Image
import torch
import torchvision.transforms.functional as F
import pyiqa
import numpy as np
from tqdm import tqdm
from face_alignment.inference import inference_face, load_pretrained_model

IMG_EXTS = (".png", ".jpg", ".jpeg")

# ===============================
# Identity tolerance (relative)
# ===============================
DELTA_ID = 0.05   # allowed drop from GFPGAN identity


# ===============================
# Utilities
# ===============================

def list_images(root):
    return {
        f for f in os.listdir(root)
        if f.lower().endswith(IMG_EXTS)
    }


def load_image(path):
    return Image.open(path).convert("RGB")


# ===============================
# Stage-6 Scorer
# ===============================

class Stage6Scorer:
    def __init__(self, device):
        self.device = device

        # IQA metrics (NR, lightweight)
        self.metrics = {
            "niqe": pyiqa.create_metric("niqe", device=device),
            "clipiqa": pyiqa.create_metric("clipiqa", device=device),
            "maniqa": pyiqa.create_metric("maniqa", device=device),
            "musiq": pyiqa.create_metric("musiq", device=device),
        }

        # AdaFace for identity
        self.adaface = load_pretrained_model("ir_50").to(device)
        self.adaface.eval()

    @torch.no_grad()
    def compute_iqa(self, img):
        tensor = F.to_tensor(img).unsqueeze(0).to(self.device)
        return {
            "NIQE": self.metrics["niqe"](tensor).item(),
            "CLIP_IQA": self.metrics["clipiqa"](tensor).item(),
            "MANIQA": self.metrics["maniqa"](tensor).item(),
            "MUSIQ": self.metrics["musiq"](tensor).item(),
        }

    @torch.no_grad()
    def compute_id(self, lq_path, pred_path):
        sim, _ = None, None
        sim, _ = inference_face(
            model=self.adaface,
            lq_path=lq_path,
            pred_path=pred_path,
            device=self.device,
            mode="test",
        )
        return sim

    def weighted_score(self, iqa):
        return (
            iqa["CLIP_IQA"]
            + iqa["MANIQA"]
            + iqa["MUSIQ"] / 100.0
            + max(0.0, (10.0 - iqa["NIQE"]) / 10.0)
        )


# ===============================
# Stage-6 execution
# ===============================

def execute(
    lq_dir,
    gfpgan_aligned,
    gfpgan_raw,
    stage5_out,
    out_dir,
    csv_out,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer = Stage6Scorer(device)

    os.makedirs(out_dir, exist_ok=True)

    lq_images = list_images(lq_dir)

    rows = []

    for name in tqdm(sorted(lq_images), desc="Stage-6 selecting"):
        lq_path = os.path.join(lq_dir, name)

        # ---------- locate GFPGAN baseline ----------
        gfpgan_path = None
        for root in [
            os.path.join(gfpgan_aligned, "restored_faces"),
            os.path.join(gfpgan_raw, "restored_imgs"),
        ]:
            if os.path.isdir(root):
                for f in os.listdir(root):
                    if f.startswith(os.path.splitext(name)[0]):
                        gfpgan_path = os.path.join(root, f)
                        break
            if gfpgan_path:
                break

        if gfpgan_path is None:
            print(f"[WARN] Missing GFPGAN baseline for {name}, skipping")
            continue

        # ---------- candidates ----------
        candidates = {
            "gfpgan": gfpgan_path,
        }

        stage5_img = os.path.join(stage5_out, name)
        if os.path.exists(stage5_img):
            candidates["stage5"] = stage5_img

        # ---------- baseline identity ----------
        id_base = scorer.compute_id(lq_path, gfpgan_path)

        best_score = -1e9
        best_key = "gfpgan"
        best_path = gfpgan_path
        best_metrics = None
        best_id = id_base

        # ---------- evaluate ----------
        for key, path in candidates.items():
            img = load_image(path)

            id_sim = scorer.compute_id(lq_path, path)
            if id_sim < id_base - DELTA_ID:
                continue

            iqa = scorer.compute_iqa(img)
            score = scorer.weighted_score(iqa)

            if score > best_score:
                best_score = score
                best_key = key
                best_path = path
                best_metrics = iqa
                best_id = id_sim
        
        torch.cuda.empty_cache()

        # ---------- save ----------
        out_path = os.path.join(out_dir, name)
        Image.open(best_path).save(out_path)

        row = {
            "image": name,
            "selected": best_key,
            "score": best_score,
            "ID_sim": best_id,
        }
        if best_metrics:
            row.update(best_metrics)

        rows.append(row)

    # ---------- CSV ----------
    if rows:
        with open(csv_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print("[Stage-6] Selection completed")


# ===============================
# CLI
# ===============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Stage-6 Final Selection")
    parser.add_argument("--lq_dir", required=True)
    parser.add_argument("--gfpgan_aligned", required=True)
    parser.add_argument("--gfpgan_raw", required=True)
    parser.add_argument("--stage5_out", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--csv_out", required=True)

    args = parser.parse_args()

    execute(
        args.lq_dir,
        args.gfpgan_aligned,
        args.gfpgan_raw,
        args.stage5_out,
        args.out_dir,
        args.csv_out,
    )
