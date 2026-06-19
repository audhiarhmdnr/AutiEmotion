"""
Arsitektur Model untuk Deteksi Emosi ASD (6 Kelas).
Menggunakan Transfer Learning dengan EfficientNet-B0.

Kelas: Natural, anger, fear, joy, sadness, surprise
"""

import torch
import torch.nn as nn
from torchvision import models


class EmotionClassifier(nn.Module):
    """
    Model klasifikasi emosi berbasis EfficientNet-B0 (6 kelas).

    EfficientNet-B0 dipilih karena:
    - Akurasi tinggi dengan ukuran parameter efisien (~5.3M).
    - Compound scaling yang efisien (resolusi + depth + width).
    - Pre-trained di ImageNet sehingga memiliki representasi fitur visual yang kuat.
    - Cocok untuk dataset kecil karena sudah memiliki fitur yang baik.
    """

    def __init__(self, num_classes=6, dropout_rate=0.5, pretrained=True):
        super(EmotionClassifier, self).__init__()

        # Load EfficientNet-B0 pre-trained
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Freeze early layers (fitur level rendah — edge, texture)
        # EfficientNet-B0 punya 8 blok features (0-7)
        # Freeze hanya 4 layer pertama untuk lebih banyak fine-tuning
        for param in self.backbone.features[:4].parameters():
            param.requires_grad = False

        # Ganti classifier head — diperkuat dengan intermediate layer
        in_features = self.backbone.classifier[1].in_features  # 1280 untuk B0
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout_rate * 0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_all(self):
        """Buka semua layer untuk fine-tuning penuh."""
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_params(self):
        """Hitung jumlah parameter yang bisa dilatih."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self):
        """Hitung total parameter."""
        return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
    model = EmotionClassifier(num_classes=6)
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")

    # Test forward pass
    dummy = torch.randn(4, 3, 224, 224)
    output = model(dummy)
    print(f"Output shape: {output.shape}")  # Harus [4, 6]
