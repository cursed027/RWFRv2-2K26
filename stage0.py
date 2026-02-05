import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import warnings
warnings.filterwarnings("ignore")



def parse_args():
    parser = argparse.ArgumentParser("Stage-0 Face Alignment")
    parser.add_argument("--input", required=True, help="Input image root")
    parser.add_argument("--output", required=True, help="Output aligned root")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--det_size", type=int, default=640)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def landmark106_to_5(lmk106):
    idx = [38, 88, 86, 52, 61]
    return lmk106[idx].astype(np.float32)


def bbox_crop(img, bbox, out_size=112, pad_ratio=0.12):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    bw, bh = x2 - x1, y2 - y1
    dx, dy = int(bw * pad_ratio), int(bh * pad_ratio)

    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w - 1, x2 + dx)
    y2 = min(h - 1, y2 + dy)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def face_score(face, img_area):
    x1, y1, x2, y2 = face.bbox
    area = (x2 - x1) * (y2 - y1)
    return 0.6 * (area / img_area) + 0.4 * face.det_score


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection", "landmark_2d_106"],
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    total = tier1 = tier2 = fail = 0

    for root, _, files in os.walk(args.input):
        rel = os.path.relpath(root, args.input)
        rel = "" if rel == "." else rel
        out_dir = os.path.join(args.output, rel)
        os.makedirs(out_dir, exist_ok=True)

        for fname in tqdm(files, disable=args.quiet):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            total += 1
            in_path = os.path.join(root, fname)
            out_path = os.path.join(out_dir, fname)

            img = cv2.imread(in_path)
            if img is None:
                fail += 1
                continue

            faces = app.get(img)
            if not faces:
                fail += 1
                continue

            img_area = img.shape[0] * img.shape[1]

            # Tier-1
            valid = []
            for f in faces:
                x1, y1, x2, y2 = f.bbox
                if (x2 - x1) * (y2 - y1) < 40 * 40:
                    continue
                if f.landmark_2d_106 is None:
                    continue
                if f.pose is not None and len(f.pose) >= 2:
                    if abs(float(f.pose[1])) > 75:
                        continue
                valid.append(f)

            if valid:
                f = max(valid, key=lambda x: face_score(x, img_area))
                kps5 = landmark106_to_5(f.landmark_2d_106)
                aligned = face_align.norm_crop(img, kps5, image_size=args.size)
                cv2.imwrite(out_path, aligned)
                tier1 += 1
                continue

            # Tier-2
            valid = []
            for f in faces:
                x1, y1, x2, y2 = f.bbox
                if (x2 - x1) * (y2 - y1) < 28 * 28:
                    continue
                valid.append(f)

            if valid:
                f = max(valid, key=lambda x: face_score(x, img_area))
                cropped = bbox_crop(img, f.bbox, args.size)
                if cropped is not None:
                    cv2.imwrite(out_path, cropped)
                    tier2 += 1
                    continue

            fail += 1

    print(f"[Stage-0] total={total} tier1={tier1} tier2={tier2} fail={fail}")

    if total == 0:
        raise RuntimeError("Stage-0 received zero images.")


if __name__ == "__main__":
    main()
