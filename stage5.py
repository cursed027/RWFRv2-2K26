import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import csv
import argparse
import subprocess
import shutil

IMG_EXTS = (".jpg", ".png", ".jpeg")
DISABLE_DIFFBIR = False # TEMPORARY: set False on new GPU

# ============================================================
# DiffBIR PRESETS (THE REAL FIX)
# ============================================================

DIFFBIR_PRESETS = {
    "aggressive": dict(cfg=4.0, steps=50, strength=1.0),
    "strong":     dict(cfg=3.5, steps=40, strength=0.8),
    "light":      dict(cfg=3.0, steps=30, strength=0.6),
    "minimal":    dict(cfg=2.5, steps=20, strength=0.5),
}

# ============================================================
# Anchor resolution
# ============================================================

def find_anchor(name, anchor_type, aligned_root, raw_root):
    if anchor_type == "aligned":
        root = os.path.join(aligned_root, "restored_faces")
    elif anchor_type == "raw":
        root = os.path.join(raw_root, "restored_imgs")
    else:
        return None

    if not os.path.isdir(root):
        return None

    stem = os.path.splitext(name)[0]
    for f in os.listdir(root):
        if f.startswith(stem) and f.lower().endswith(IMG_EXTS):
            return os.path.join(root, f)
    return None


def find_latest_image(root):
    exts = (".png", ".jpg", ".jpeg")
    candidates = []

    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(exts):
                candidates.append(os.path.join(r, f))

    if not candidates:
        return None

    candidates.sort(key=lambda x: os.path.getmtime(x))
    return candidates[-1]


# ============================================================
# DiffBIR execution (REAL CLI)
# ============================================================

def run_diffbir(input_img, output_dir, preset):
    # Always start clean
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Create a fresh input folder
    input_dir = os.path.join(output_dir, "_input")
    os.makedirs(input_dir, exist_ok=True)

    # Copy image ONLY if source and destination differ
    img_name = os.path.basename(input_img)
    tmp_input_img = os.path.join(input_dir, img_name)

    if os.path.realpath(input_img) != os.path.realpath(tmp_input_img):
        shutil.copy(input_img, tmp_input_img)

    # Load preset
    p = DIFFBIR_PRESETS[preset]

    diffbir_root = os.path.abspath("third_party/DiffBIR")

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    cmd = [
        "python", "-u", "inference.py",
        "--task", "face",
        "--version", "v2",
        "--sampler", "spaced",
        "--steps", str(p["steps"]),
        "--cfg_scale", str(p["cfg"]),
        "--strength", str(p["strength"]),
        "--captioner", "none",
        "--pos_prompt", "",
        "--neg_prompt", "low quality, blurry, low-resolution, noisy, unsharp, weird textures",
        "--input", input_dir,      # ABSOLUTE
        "--output", output_dir,    # ABSOLUTE
        "--device", "cuda",
        "--precision", "fp16"
    ]

    subprocess.run(cmd, cwd=diffbir_root, check=True)


    out = find_latest_image(output_dir)
    assert out is not None and out.lower().endswith(IMG_EXTS), "DiffBIR produced no image output"

    if out is None:
        raise RuntimeError("DiffBIR produced no output")

    return out

# ============================================================
# CodeFormer (ALIGNED ONLY)
# ============================================================

def run_codeformer(input_img, output_dir, weight):
    """
    input_img: path to a SINGLE aligned face image
    output_dir: directory where CodeFormer will write results
    """

    # HARD RESET per image
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(input_img):
        raise FileNotFoundError(f"CodeFormer input image not found: {input_img}")

    cmd = [
        "python", "inference_codeformer.py",
        "--has_aligned",
        "-w", str(weight),
        "--input_path", os.path.abspath(input_img),
        "--output_path", os.path.abspath(output_dir)
    ]

    subprocess.run(cmd, cwd="third_party/CodeFormer", check=True)

    # 🔴 THIS IS THE IMPORTANT PART
    restored_dir = os.path.join(output_dir, "restored_faces")
    if not os.path.isdir(restored_dir):
        raise RuntimeError("CodeFormer did not create restored_faces directory")

    out = find_latest_image(restored_dir)
    if out is None:
        raise RuntimeError("CodeFormer restored_faces is empty")

    return out


# ============================================================
# Stage-5 execution
# ============================================================

def execute(routes_csv, gfpgan_aligned, gfpgan_raw, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(routes_csv) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        name = r["image"]
        actions = r["actions"].split(";")
        anchor_type = r["anchor_type"]

        anchor = find_anchor(name, anchor_type, gfpgan_aligned, gfpgan_raw)
        if anchor is None:
            print(f"[WARN] Missing anchor: {name}")
            continue

        geometry = "aligned" if anchor_type == "aligned" else "unaligned"
        current = anchor

        for act in actions:
            act = act.strip()
            #print(f"DEBUG action = [{act}]")


            if act == "use_anchor":
                continue

            if act.startswith("diffbir"):
                if DISABLE_DIFFBIR:
                    # Skip DiffBIR, keep current anchor
                    continue
                preset = {
                    "diffbir_0.35": "aggressive",   # Case A
                    "diffbir_0.25": "strong",       # Case B
                    "diffbir_0.15": "light",        # Case D1
                    "diffbir_0.1":  "minimal",      # Case C / D2
                }[act]


                tmp = os.path.join(out_dir, f"_tmp_diffbir_{os.path.splitext(name)[0]}")
                current = run_diffbir(current, tmp, preset)


            elif act.startswith("codeformer"):
                if geometry != "aligned":
                    continue
                w = float(act.split("_")[1])
                tmp = os.path.join(out_dir, f"_tmp_codeformer_{os.path.splitext(name)[0]}")
                current = run_codeformer(current, tmp, w)

        shutil.copy(current, os.path.join(out_dir, name))

    print("[Stage-5] Execution completed")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Stage-5 Execution (Correct DiffBIR)")
    parser.add_argument("--routes_csv", required=True)
    parser.add_argument("--gfpgan_aligned", required=True)
    parser.add_argument("--gfpgan_raw", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    execute(
        args.routes_csv,
        args.gfpgan_aligned,
        args.gfpgan_raw,
        args.out_dir
    )
