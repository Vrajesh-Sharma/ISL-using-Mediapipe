"""
data_prep/audit_no_detect.py

MediaPipe will occasionally miss the hand (heavy motion blur, unusual crop,
extreme lighting, hand partly out of frame). This lists/saves those cases so
you can decide whether to: (a) lower MP_MIN_DETECTION_CONFIDENCE, (b) drop
them from training, or (c) leave them — the model still gets the full image
as a fallback crop and a zero landmark vector, so the image branch alone can
still carry these examples.

Usage:
    python data_prep/audit_no_detect.py --split train
"""
import argparse
import csv
import os
import shutil
import config


def main(split, copy_to=None):
    prefix = os.path.join(config.CACHE_DIR, split)
    with open(prefix + "_manifest.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    missing = [r for r in rows if int(r["num_hands"]) == 0]
    print(f"[{split}] {len(missing)}/{len(rows)} images ({100*len(missing)/len(rows):.1f}%) had no hand detected")

    by_class = {}
    for r in missing:
        by_class.setdefault(r["label"] or "(unlabeled)", 0)
        by_class[r["label"] or "(unlabeled)"] += 1
    for cls, n in sorted(by_class.items(), key=lambda kv: -kv[1]):
        print(f"  {cls}: {n}")

    if copy_to:
        os.makedirs(copy_to, exist_ok=True)
        for r in missing:
            shutil.copy(r["filepath"], os.path.join(copy_to, os.path.basename(r["filepath"])))
        print(f"copied {len(missing)} files to {copy_to} for manual inspection")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--copy_to", default=None)
    args = ap.parse_args()
    main(args.split, args.copy_to)
