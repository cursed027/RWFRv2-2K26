import warnings
warnings.filterwarnings("ignore", message=".*functional_tensor.*")
warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*")
warnings.filterwarnings("ignore", message=".*Arguments other than a weight enum.*")

import os
import subprocess
import argparse

IMG_EXTS = (".jpg", ".png", ".jpeg")


def has_images(root):
    for _, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                return True
    return False


def count_gfpgan_outputs(root, aligned):
    sub = "restored_faces" if aligned else "restored_imgs"
    target = os.path.join(root, sub)

    if not os.path.isdir(target):
        return 0

    count = 0
    for r, _, files in os.walk(target):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                count += 1
    return count

def collect_images(root):
    imgs = set()
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                imgs.add(f)
    return imgs


def copy_subset(src_root, dst_root, filenames):
    os.makedirs(dst_root, exist_ok=True)
    for r, _, files in os.walk(src_root):
        for f in files:
            if f in filenames:
                src = os.path.join(r, f)
                dst = os.path.join(dst_root, f)
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(src), dst)

def run_stage0(input_dir, aligned_dir):
    subprocess.run(
        [
            "python", "stage0.py",
            "--input", input_dir,
            "--output", aligned_dir,
            "--quiet"
        ],
        check=True
    )
    
def run_gfpgan_aligned(input_dir, output_dir):
    cmd = [
        "python", "inference_gfpgan.py",
        "-i", os.path.abspath(input_dir),
        "-o", os.path.abspath(output_dir),
        "-v", "1.4",
        "-s", "1",
        "--aligned",
        "--only_center_face",
        "--bg_upsampler", "none"
    ]
    subprocess.run(cmd, cwd="third_party/GFPGAN", check=True)



def run_gfpgan_raw(input_dir, output_dir):
    cmd = [
        "python", "inference_gfpgan.py",
        "-i", os.path.abspath(input_dir),
        "-o", os.path.abspath(output_dir),
        "-v", "1.4",
        "-s", "1",
        "--bg_upsampler", "none"
    ]
    subprocess.run(cmd, cwd="third_party/GFPGAN", check=True)


def main():
    parser = argparse.ArgumentParser("NTIRE Face Restoration Pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--workdir", default="work")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute Stage-4 routes (DiffBIR / CodeFormer)"
    )
    parser.add_argument(
        "--final_out",
        default=None,
        help="Output directory for final restored images"
    )
    parser.add_argument(
    "--select",
    action="store_true",
    help="Execute Stage-6 final selection (IQA + AdaFace)"
    )


    args = parser.parse_args()

    stage0_out = os.path.join(args.workdir, "stage0_aligned")
    stage1_aligned = os.path.join(args.workdir, "stage1_gfpgan_aligned")
    stage1_raw = os.path.join(args.workdir, "stage1_gfpgan_raw")
    
    stage2_csv = os.path.join(args.workdir, "stage2_metrics.csv")
    stage3_csv = os.path.join(args.workdir, "stage3_thresholds.csv")
    stage4_csv = os.path.join(args.workdir, "stage4_routes.csv")
    stage6_out = os.path.join(args.workdir, "stage6_selected")
    stage6_csv = os.path.join(args.workdir, "stage6_metrics.csv")

    os.makedirs(stage0_out, exist_ok=True)
    os.makedirs(stage1_aligned, exist_ok=True)
    os.makedirs(stage1_raw, exist_ok=True)

    # -------- Stage-0 --------
    run_stage0(args.input, stage0_out)

    all_inputs = collect_images(args.input)
    aligned_ok = collect_images(stage0_out)
    aligned_fail = all_inputs - aligned_ok

    print(f"[Stage-1] aligned_ok={len(aligned_ok)} aligned_fail={len(aligned_fail)}")

    # -------- Stage-1A: aligned GFPGAN --------
    if aligned_ok:
        run_gfpgan_aligned(stage0_out, stage1_aligned)
        print(f"[Stage-1] GFPGAN aligned outputs: {count_gfpgan_outputs(stage1_aligned, aligned=True)}")

    # -------- Stage-1B: raw GFPGAN ONLY for failures --------
    if aligned_fail:
        raw_input_subset = os.path.join(args.workdir, "raw_only_inputs")
        copy_subset(args.input, raw_input_subset, aligned_fail)

        run_gfpgan_raw(raw_input_subset, stage1_raw)
        print(f"[Stage-1] GFPGAN raw outputs: {count_gfpgan_outputs(stage1_raw, aligned=False)}")

    
    # -------- Stage-2 --------
    subprocess.run(
        [
            "python", "stage2.py",
            "--stage0", stage0_out,
            "--gfpgan_aligned", stage1_aligned,
            "--gfpgan_raw", stage1_raw,
            "--out", stage2_csv
        ],
        check=True
    )

    # -------- Stage-3 --------
    subprocess.run(
        [
            "python", "stage3.py",
            "--in_csv", stage2_csv,
            "--out_csv", stage3_csv
        ],
        check=True
    )

    # -------- Stage-4 --------
    subprocess.run(
        [
            "python", "stage4.py",
            "--in_csv", stage3_csv,
            "--out_csv", stage4_csv
        ],
        check=True
    )

    print("[Pipeline] Completed Stage-4 routing")
    print(f"[Pipeline] Routing table: {stage4_csv}")
    
    # -------- Stage-5(4.5) (optional execution) --------
    if args.execute:
        if args.final_out is None:
            raise ValueError("--final_out must be specified when --execute is used")

        subprocess.run(
            [
                "python", "stage5.py",
                "--routes_csv", stage4_csv,
                "--gfpgan_aligned", stage1_aligned,
                "--gfpgan_raw", stage1_raw,
                "--out_dir", args.final_out
            ],
            check=True
        )

    if args.execute:
        print("[Pipeline] Stage-5 execution completed")
    else:
        print("[Pipeline] Stage-5 execution skipped (routing only)")
        
    # -------- Stage-6 (final selection) --------
    if args.select:
        subprocess.run(
            [
                "python", "stage6.py",
                "--lq_dir", args.input,
                "--gfpgan_aligned", stage1_aligned,
                "--gfpgan_raw", stage1_raw,
                "--stage5_out", args.final_out,
                "--out_dir", stage6_out,
                "--csv_out", stage6_csv,
            ],
            check=True
        )
        print("[Pipeline] Stage-6 selection completed")
    else:
        print("[Pipeline] Stage-6 skipped")



if __name__ == "__main__":
    main()
