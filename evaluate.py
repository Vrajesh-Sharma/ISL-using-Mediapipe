"""
evaluate.py

Loads the best checkpoint and reports:
  - accuracy, macro/weighted precision/recall/F1
  - full per-class classification report
  - confusion matrix (saved as PNG heatmap + CSV)
  - a focused confusion sub-report for visually-similar class groups
    (config.CONFUSABLE_GROUPS)

Run once on the held-out val split (sanity check) and, more importantly, on
the 36k-image unseen TEST_DIR cache — this second run is the number that
actually matters for the project's stated goal.

Usage:
    python data_prep/extract_landmarks.py --data_dir /path/to/isl_test_unseen --split test
    python evaluate.py --split test
    python evaluate.py --split val
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import config
from data_prep.dataset import load_manifest, ISLDataset, stratified_train_val_split
from models.fusion_model import FusionISLModel


def get_loader_for_split(split):
    if split == "val":
        rows, landmarks = load_manifest("train")
        _, val_idx = stratified_train_val_split(rows, config.VAL_SPLIT, config.SEED)
        ds = ISLDataset(rows, landmarks, val_idx, train=False)
    else:
        rows, landmarks = load_manifest(split)
        all_idx = list(range(len(rows)))
        ds = ISLDataset(rows, landmarks, all_idx, train=False)
    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False,
                         num_workers=config.NUM_WORKERS, pin_memory=True)
    return loader


def load_model(device):
    ckpt = torch.load(os.path.join(config.CHECKPOINT_DIR, "best_model.pt"), map_location=device)
    model = FusionISLModel().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded checkpoint from epoch {ckpt['epoch']} (val macro-F1={ckpt['val_f1']:.4f})")
    return model


@torch.no_grad()
def run_inference(model, loader, device):
    all_preds, all_labels, all_probs = [], [], []
    for img, lm, label in loader:
        img, lm = img.to(device), lm.to(device)
        probs = model.predict_proba(img, lm)
        preds = probs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(label.numpy().tolist())
        all_probs.append(probs.cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.concatenate(all_probs, axis=0)


def confusable_report(y_true, y_pred):
    print("\n=== Focused confusion among visually-similar groups ===")
    for group in config.CONFUSABLE_GROUPS:
        idxs = [config.CLASS_TO_IDX[c] for c in group if c in config.CLASS_TO_IDX]
        mask = np.isin(y_true, idxs)
        if mask.sum() == 0:
            continue
        sub_true, sub_pred = y_true[mask], y_pred[mask]
        acc = accuracy_score(sub_true, sub_pred)
        print(f"group {group}: n={mask.sum()}, in-group accuracy={acc:.4f}")
        cm = confusion_matrix(sub_true, sub_pred, labels=idxs)
        print("  confusion (rows=true, cols=pred, order=" + str(group) + "):")
        print(" ", cm)


def main(split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    loader = get_loader_for_split(split)

    y_true, y_pred, y_prob = run_inference(model, loader, device)
    valid = y_true >= 0  # drop any rows without labels (shouldn't occur for train/val/labeled test)
    y_true, y_pred = y_true[valid], y_pred[valid]

    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== Split: {split} | N={len(y_true)} | Accuracy={acc:.4f} ===")
    report = classification_report(y_true, y_pred, labels=list(range(config.NUM_CLASSES)),
                                    target_names=config.CLASSES, digits=4, zero_division=0)
    print(report)
    with open(os.path.join(config.RESULTS_DIR, f"{split}_classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n" + report)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    np.savetxt(os.path.join(config.RESULTS_DIR, f"{split}_confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="viridis", xticklabels=config.CLASSES, yticklabels=config.CLASSES)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion matrix ({split}, acc={acc:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_DIR, f"{split}_confusion_matrix.png"), dpi=150)
    plt.close()

    confusable_report(y_true, y_pred)
    print(f"\nSaved report/CSV/PNG under {config.RESULTS_DIR}/{split}_*")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()
    main(args.split)
