"""
models/fusion_model.py

Two branches, fused before the final classifier:
  IMAGE BRANCH   : pretrained lightweight CNN (MobileNetV3-Small/Large or
                   EfficientNet-B0) on the cropped hand image -> appearance features
  LANDMARK BRANCH: small MLP on the 179-d MediaPipe-derived geometric feature
                   vector -> explicit hand-structure features

Fusion: concat -> MLP -> 36-way logits. A learned gate additionally lets the
model down-weight the image branch when its own landmark confidence is high
and vice versa (helps when MediaPipe fails to find a hand on a hard test image
and the landmark branch is effectively all zeros).
"""
import torch
import torch.nn as nn
import torchvision.models as tvm

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def build_backbone(name):
    if name == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        feat_dim = m.classifier[0].in_features
        m.classifier = nn.Identity()
        return m, feat_dim
    if name == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        feat_dim = m.classifier[0].in_features
        m.classifier = nn.Identity()
        return m, feat_dim
    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feat_dim = m.classifier[1].in_features
        m.classifier = nn.Identity()
        return m, feat_dim
    raise ValueError(f"unknown backbone {name}")


class LandmarkBranch(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class FusionISLModel(nn.Module):
    def __init__(self,
                 num_classes=config.NUM_CLASSES,
                 landmark_dim=config.LANDMARK_FEATURE_DIM,
                 backbone_name=config.IMAGE_BACKBONE,
                 img_embed_dim=config.IMG_EMBED_DIM,
                 lm_embed_dim=config.LM_EMBED_DIM,
                 fusion_hidden=config.FUSION_HIDDEN_DIM,
                 dropout=config.DROPOUT):
        super().__init__()
        self.backbone, backbone_feat_dim = build_backbone(backbone_name)
        self.img_proj = nn.Sequential(
            nn.Linear(backbone_feat_dim, img_embed_dim), nn.BatchNorm1d(img_embed_dim),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
        )
        self.lm_branch = LandmarkBranch(landmark_dim, lm_embed_dim, dropout)

        # learned scalar gate per branch, conditioned on both embeddings
        self.gate = nn.Sequential(
            nn.Linear(img_embed_dim + lm_embed_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2), nn.Softmax(dim=-1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(img_embed_dim + lm_embed_dim, fusion_hidden), nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def set_backbone_trainable(self, trainable: bool):
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def forward(self, img, lm):
        img_feat = self.backbone(img)
        if img_feat.ndim > 2:
            img_feat = torch.flatten(img_feat, 1)
        img_emb = self.img_proj(img_feat)
        lm_emb = self.lm_branch(lm)

        gates = self.gate(torch.cat([img_emb, lm_emb], dim=1))  # (B,2)
        img_g = gates[:, 0:1]
        lm_g = gates[:, 1:2]
        fused = torch.cat([img_emb * img_g, lm_emb * lm_g], dim=1)

        logits = self.classifier(fused)
        return logits

    @torch.no_grad()
    def predict_proba(self, img, lm):
        logits = self.forward(img, lm)
        return torch.softmax(logits, dim=1)
