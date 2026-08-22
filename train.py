"""
train.py

Trains the fusion model on the CACHED train-split landmarks/manifest
(produced by data_prep/extract_landmarks.py --split train).
The 36k-image TEST_DIR is NEVER touched here — see evaluate.py for that.

Usage:
    python data_prep/extract_landmarks.py --data_dir /path/to/isl_train --split train
    python train.py
"""
import os
import time
import math
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import config
from data_prep.dataset import load_manifest, ISLDataset, stratified_train_val_split, class_counts
from models.fusion_model import FusionISLModel

# Fraction of TRAINING landmark vectors to randomly zero-out per batch, to
# simulate MediaPipe detection failure at test time. Your own audit showed:
#   train no-detect rate ≈ 1.0%   (542/54000)
#   test  no-detect rate ≈ 36.4%  (13111/36000)
# Without this, the model almost never learns to cope with an all-zero
# landmark vector during training, then hits it on a third of the real test
# set. Set to match (or slightly exceed) your observed TEST no-detect rate.
# Falls back to 0.35 if not set in config.py.
LANDMARK_DROPOUT_PROB = getattr(config, "LANDMARK_DROPOUT_PROB", 0.35)


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def make_sampler(rows, indices):
    counts = class_counts(rows, indices)
    counts = np.clip(counts, 1, None)
    class_weight = 1.0 / counts
    sample_weights = np.array([class_weight[config.CLASS_TO_IDX[rows[i]["label"]]] for i in indices])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def cosine_warmup_lr(optimizer, step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        lr = base_lr * (step + 1) / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g["lr"] = lr
    return lr


def apply_landmark_dropout(lm, p):
    """Zero out entire landmark vectors for a random subset of the batch,
    matching the all-zero vector extract_landmarks.py produces when
    MediaPipe finds no hand. Train-only — val/test keep real landmarks."""
    if p <= 0:
        return lm
    mask = (torch.rand(lm.size(0), device=lm.device) < p).float().unsqueeze(1)  # (B,1)
    return lm * (1.0 - mask)


def run_epoch(model, loader, criterion, optimizer, device, scaler, train, epoch_num=None, total_epochs=None,
              step_counter=None, total_steps=None, warmup_steps=None, base_lr=None):
    model.train(mode=train)
    total_loss, all_preds, all_labels = 0.0, [], []

    desc = f"{'train' if train else 'val  '} epoch {epoch_num}/{total_epochs}" if epoch_num else ("train" if train else "val")
    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)

    amp_enabled = config.MIXED_PRECISION and device.type == "cuda"
    for img, lm, label in pbar:
        img, lm, label = img.to(device, non_blocking=True), lm.to(device, non_blocking=True), label.to(device, non_blocking=True)

        if train:
            lm = apply_landmark_dropout(lm, LANDMARK_DROPOUT_PROB)
            step_counter[0] += 1
            lr = cosine_warmup_lr(optimizer, step_counter[0], total_steps, warmup_steps, base_lr)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(img, lm)
                loss = criterion(logits, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            lr = None
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(img, lm)
                loss = criterion(logits, label)

        total_loss += loss.item() * img.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(label.detach().cpu().numpy().tolist())

        running_acc = accuracy_score(all_labels, all_preds)
        postfix = {"loss": f"{total_loss/len(all_labels):.4f}", "acc": f"{running_acc:.4f}"}
        if lr is not None:
            postfix["lr"] = f"{lr:.2e}"
        pbar.set_postfix(postfix)

    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1


def main():
    set_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print(f"landmark dropout (train-time, simulating MediaPipe misses): p={LANDMARK_DROPOUT_PROB}")

    rows, landmarks = load_manifest("train")
    train_idx, val_idx = stratified_train_val_split(rows, config.VAL_SPLIT, config.SEED)
    print(f"train={len(train_idx)} val={len(val_idx)}")

    train_ds = ISLDataset(rows, landmarks, train_idx, train=True)
    val_ds = ISLDataset(rows, landmarks, val_idx, train=False)

    sampler = make_sampler(rows, train_idx) if config.USE_CLASS_BALANCED_SAMPLER else None
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler,
                               shuffle=(sampler is None), num_workers=config.NUM_WORKERS,
                               pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                             num_workers=config.NUM_WORKERS, pin_memory=True)

    model = FusionISLModel().to(device)
    model.set_backbone_trainable(False)  # start with backbone frozen (fast head warmup)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.BASE_LR, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device.type, enabled=(config.MIXED_PRECISION and device.type == "cuda"))

    total_steps = config.EPOCHS * len(train_loader)
    warmup_steps = config.WARMUP_EPOCHS * len(train_loader)
    step_counter = [0]

    best_val_f1 = -1.0
    epochs_no_improve = 0
    history = []

    epoch_bar = tqdm(range(config.EPOCHS), desc="overall", position=0)
    for epoch in epoch_bar:
        if epoch == config.FREEZE_BACKBONE_EPOCHS:
            tqdm.write(">> unfreezing CNN backbone")
            model.set_backbone_trainable(True)

        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, optimizer, device, scaler,
                                            train=True, epoch_num=epoch + 1, total_epochs=config.EPOCHS,
                                            step_counter=step_counter, total_steps=total_steps,
                                            warmup_steps=warmup_steps, base_lr=config.BASE_LR)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, scaler,
                                               train=False, epoch_num=epoch + 1, total_epochs=config.EPOCHS)
        dt = time.time() - t0

        tqdm.write(f"epoch {epoch+1}/{config.EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.4f} f1 {tr_f1:.4f} "
                   f"| val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f} | {dt:.1f}s")
        history.append({"epoch": epoch + 1, "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
                         "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1})
        epoch_bar.set_postfix({"best_val_f1": f"{best_val_f1:.4f}"})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "val_f1": val_f1,
                        "config_snapshot": {k: v for k, v in vars(config).items() if k.isupper()}},
                       ckpt_path)
            tqdm.write(f"  -> new best (val macro-F1={val_f1:.4f}), saved to {ckpt_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.EARLY_STOP_PATIENCE:
                tqdm.write(f"Early stopping at epoch {epoch+1} (no val improvement for {config.EARLY_STOP_PATIENCE} epochs)")
                break

    with open(os.path.join(config.RESULTS_DIR, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best val macro-F1: {best_val_f1:.4f}. History saved to {config.RESULTS_DIR}/train_history.json")


if __name__ == "__main__":
    main()