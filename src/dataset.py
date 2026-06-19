"""
Dataset & DataLoader untuk Deteksi Emosi ASD.
Menggunakan PyTorch Dataset + torchvision transforms untuk augmentasi.

Klasifikasi 6 Kelas:
  - Natural (0), anger (1), fear (2), joy (3), sadness (4), surprise (5)
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# ===================== KONFIGURASI KELAS =====================
# 6 kelas emosi asli
CLASS_NAMES = ['Natural', 'anger', 'fear', 'joy', 'sadness', 'surprise']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# Label yang lebih mudah dibaca (untuk display)
DISPLAY_LABELS = {
    'Natural': 'Natural',
    'anger': 'Anger',
    'fear': 'Fear',
    'joy': 'Joy',
    'sadness': 'Sadness',
    'surprise': 'Surprise',
}

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# =============================================================


def get_train_transforms():
    """Augmentasi untuk data training — moderat karena sudah ada offline augmentation."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(IMG_SIZE + 32),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def get_val_transforms():
    """Transform untuk data validasi/test (tanpa augmentasi)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transforms():
    """
    Test-Time Augmentation (TTA) transforms.
    Menghasilkan beberapa variasi dari gambar yang sama untuk di-rata-ratakan saat inference.
    """
    return [
        # Original
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        # Slight brightness change
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    ]


class EmotionDataset(Dataset):
    """Dataset untuk gambar emosi anak ASD (6 kelas)."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path ke folder (train atau test) yang berisi subfolder per kelas.
            transform: torchvision transform pipeline.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for class_name in CLASS_NAMES:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"[WARNING] Folder kelas '{class_name}' tidak ditemukan di {root_dir}")
                continue

            class_idx = CLASS_TO_IDX[class_name]
            count = 0
            for fname in os.listdir(class_dir):
                fpath = os.path.join(class_dir, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    self.samples.append((fpath, class_idx))
                    count += 1

            print(f"  {class_name} ({DISPLAY_LABELS[class_name]}): {count} gambar")

        print(f"[INFO] Total {len(self.samples)} gambar dari {root_dir}")

        # Tampilkan distribusi
        dist = {}
        for _, label in self.samples:
            name = IDX_TO_CLASS[label]
            dist[name] = dist.get(name, 0) + 1
        print(f"[INFO] Distribusi: {dist}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)

        if image is None:
            # Fallback jika gambar rusak: buat gambar hitam
            print(f"[WARNING] Gagal membaca gambar: {img_path}")
            image = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, label


def compute_class_weights(data_root):
    """Hitung class weights berdasarkan distribusi data training (6 kelas)."""
    train_dir = os.path.join(data_root, 'train')
    counts = {name: 0 for name in CLASS_NAMES}

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            n = len([f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])
            counts[class_name] = n

    total = sum(counts.values())
    num_classes = len(CLASS_NAMES)
    weights = []
    for name in CLASS_NAMES:
        w = total / (num_classes * max(counts[name], 1))
        weights.append(w)

    # Normalisasi agar rata-rata = 1
    avg_w = sum(weights) / len(weights)
    weights = [w / avg_w for w in weights]

    print(f"[INFO] Distribusi data train: {counts}")
    print(f"[INFO] Class weights: {dict(zip(CLASS_NAMES, [f'{w:.3f}' for w in weights]))}")
    return torch.FloatTensor(weights)


def get_dataloaders(data_root, batch_size=32, num_workers=0, val_split=0.15):
    """
    Membuat DataLoader untuk train, validation, dan test.

    Args:
        data_root: Path ke folder dataset (berisi subfolder 'train' dan 'test').
        batch_size: Ukuran batch.
        num_workers: Jumlah worker untuk loading data.
        val_split: Proporsi data training untuk validasi (0.0 - 1.0).

    Returns:
        train_loader, val_loader, test_loader
    """
    train_dir = os.path.join(data_root, 'train')
    test_dir = os.path.join(data_root, 'test')

    # Full training dataset
    full_train_dataset = EmotionDataset(train_dir, transform=get_train_transforms())
    test_dataset = EmotionDataset(test_dir, transform=get_val_transforms())

    # Split training menjadi train + validation
    total = len(full_train_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    # Gunakan random split tapi seed tetap agar reproducible
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Buat validation dataset dengan transform berbeda (tanpa augmentasi)
    val_dataset = EmotionDataset(train_dir, transform=get_val_transforms())

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    print(f"\n[INFO] Split: Train={train_size}, Val={val_size}, Test={len(test_dataset)}")

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Quick test — coba dengan data augmented dulu, fallback ke raw
    augmented_root = os.path.join(os.path.dirname(__file__), '..', 'data', 'augmented')
    raw_root = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'Autism emotion recogition dataset')

    data_root = augmented_root if os.path.isdir(os.path.join(augmented_root, 'train')) else raw_root
    print(f"Menggunakan data dari: {data_root}\n")

    train_loader, val_loader, test_loader = get_dataloaders(data_root, batch_size=16, num_workers=0)
    print(f"\nTrain batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Cek satu batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels: {labels}")

    # Cek class weights
    weights = compute_class_weights(data_root)
    print(f"Weights tensor: {weights}")
