"""
Skrip Training untuk Model Deteksi Emosi ASD (6 Kelas).
Mendukung:
- Focal Loss untuk fokus ke sampel sulit
- MixUp augmentation untuk regularisasi
- Training dengan early stopping
- Learning rate scheduling (Cosine Annealing + Warmup)
- Mixed Precision Training (AMP)
- Class weights untuk kelas tidak seimbang
- Label smoothing untuk regularisasi
- Gradient clipping untuk stabilitas
- Validation split dari training data
- Logging metrik ke console
- Penyimpanan model terbaik
"""

import os
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
print("Memulai train.py...", flush=True)

import time
import copy
print("  Loading torch...", flush=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast
print("  Loading sklearn...", flush=True)
from sklearn.metrics import f1_score, classification_report, confusion_matrix
print("  Loading numpy...", flush=True)
import numpy as np
print("  Imports selesai!", flush=True)

# Tambahkan parent dir ke path
sys.path.insert(0, os.path.dirname(__file__))

from dataset import get_dataloaders, CLASS_NAMES, DISPLAY_LABELS, compute_class_weights
from model import EmotionClassifier

# ======================== KONFIGURASI ========================
CONFIG = {
    'data_root': None,  # Akan ditentukan otomatis (augmented > raw)
    'save_dir': os.path.join(os.path.dirname(__file__), '..', 'models'),
    'batch_size': 32,
    'num_workers': 0,        # Set 0 di Windows untuk menghindari masalah multiprocessing
    'num_epochs': 80,        # Lebih banyak epoch untuk 6 kelas
    'learning_rate': 3e-4,   # Learning rate awal
    'weight_decay': 5e-3,    # Regularisasi kuat
    'dropout_rate': 0.5,     # Dropout tinggi untuk mencegah overfitting
    'patience': 20,          # Early stopping patience
    'unfreeze_epoch': 10,    # Epoch ke berapa mulai unfreeze semua layer
    'fine_tune_lr': 3e-5,    # Learning rate setelah unfreeze
    'label_smoothing': 0.1,  # Regularisasi pada label
    'focal_gamma': 2.0,      # Focal Loss gamma
    'mixup_alpha': 0.3,      # MixUp alpha
    'grad_clip': 1.0,        # Gradient clipping
    'val_split': 0.15,       # Proporsi data training untuk validasi
}
# =============================================================


def get_data_root():
    """Menentukan data root — prioritas: augmented > raw."""
    augmented = os.path.join(os.path.dirname(__file__), '..', 'data', 'augmented')
    raw = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'Autism emotion recogition dataset')

    if os.path.isdir(os.path.join(augmented, 'train')):
        print("[INFO] Menggunakan dataset AUGMENTED (data/augmented/)")
        return augmented
    else:
        print("[INFO] Dataset augmented tidak ditemukan, menggunakan dataset RAW.")
        print("[TIP]  Jalankan 'python src/augment_dataset.py' untuk membuat dataset augmented.")
        return raw


class FocalLoss(nn.Module):
    """
    Focal Loss — mengurangi kontribusi sampel mudah dan fokus ke sampel sulit.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def mixup_data(x, y, alpha=0.3):
    """MixUp augmentation — mencampurkan 2 gambar dan label secara proporsional."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss function untuk MixUp."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_mixup=True, mixup_alpha=0.3):
    """Melatih model untuk satu epoch dengan Mixed Precision + MixUp."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast('cuda', enabled=(device.type == 'cuda')):
            if use_mixup and np.random.random() > 0.3:
                mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
                outputs = model(mixed_images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                _, preds = torch.max(outputs, 1)
                correct += (lam * (preds == labels_a).sum().item() +
                           (1 - lam) * (preds == labels_b).sum().item())
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, epoch_acc, epoch_f1


def evaluate(model, loader, criterion, device):
    """Evaluasi model pada dataset validasi/test."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels


def main():
    print("=" * 60)
    print("  TRAINING MODEL DETEKSI EMOSI ASD")
    print("  6 Kelas: Natural, Anger, Fear, Joy, Sadness, Surprise")
    print("=" * 60)

    # Tentukan data root
    data_root = get_data_root()
    CONFIG['data_root'] = data_root

    # Device & GPU Optimization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[INFO] Menggunakan device: {device}")

    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[INFO] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True
    else:
        print("[WARNING] GPU tidak terdeteksi. Training di CPU akan jauh lebih lambat.")

    # Data
    print(f"\n[INFO] Memuat dataset dari: {data_root}")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root,
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        val_split=CONFIG['val_split']
    )

    # Hitung class weights
    class_weights = compute_class_weights(data_root).to(device)

    # Model (6 kelas)
    model = EmotionClassifier(
        num_classes=len(CLASS_NAMES),
        dropout_rate=CONFIG['dropout_rate'],
        pretrained=True
    ).to(device)
    print(f"[INFO] Total parameter: {model.get_total_params():,}")
    print(f"[INFO] Parameter trainable: {model.get_trainable_params():,}")

    # Loss function — Focal Loss
    criterion = FocalLoss(
        weight=class_weights,
        gamma=CONFIG['focal_gamma'],
        label_smoothing=CONFIG['label_smoothing']
    )
    eval_criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)

    # Mixed Precision Scaler
    scaler = GradScaler('cuda', enabled=(device.type == 'cuda'))

    # Buat folder simpan model
    os.makedirs(CONFIG['save_dir'], exist_ok=True)

    # Training Loop
    best_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"  Mulai Training - {CONFIG['num_epochs']} epochs")
    print(f"  Kelas: {CLASS_NAMES}")
    print(f"  Focal Loss gamma: {CONFIG['focal_gamma']}")
    print(f"  MixUp alpha: {CONFIG['mixup_alpha']}")
    print(f"  Label Smoothing: {CONFIG['label_smoothing']}")
    print(f"{'='*60}\n")

    start_time = time.time()

    for epoch in range(CONFIG['num_epochs']):
        # Unfreeze semua layer setelah epoch tertentu
        if epoch == CONFIG['unfreeze_epoch']:
            print(f"\n[INFO] Epoch {epoch}: Membuka semua layer untuk fine-tuning penuh!")
            model.unfreeze_all()
            print(f"[INFO] Parameter trainable sekarang: {model.get_trainable_params():,}")
            optimizer = optim.AdamW(
                model.parameters(),
                lr=CONFIG['fine_tune_lr'],
                weight_decay=CONFIG['weight_decay']
            )
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-7)
            patience_counter = 0

        use_mixup = epoch >= 3

        # Train
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            use_mixup=use_mixup, mixup_alpha=CONFIG['mixup_alpha']
        )

        # Evaluate on validation set
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, eval_criterion, device
        )

        scheduler.step()

        overfit_gap = train_acc - val_acc
        current_lr = optimizer.param_groups[0]['lr']
        mixup_str = " [MixUp]" if use_mixup else ""
        print(f"Epoch [{epoch+1:2d}/{CONFIG['num_epochs']}]{mixup_str} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
              f"Gap: {overfit_gap:.3f} LR: {current_lr:.2e}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print(f"  -> Model terbaik baru! Val F1: {best_f1:.4f}")
            
            # Langsung simpan ke disk agar tidak hilang jika di-stop paksa (CTRL+C)
            save_path = os.path.join(CONFIG['save_dir'], 'emotion_model_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': CLASS_NAMES,
                'config': CONFIG,
                'best_f1': best_f1,
            }, save_path)
            print(f"  -> Tersimpan di: {save_path}")
        else:
            patience_counter += 1

        if patience_counter >= CONFIG['patience']:
            print(f"\n[INFO] Early stopping di epoch {epoch+1} (no improvement selama {CONFIG['patience']} epochs)")
            break

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training Selesai! Waktu: {total_time:.1f} detik ({total_time/60:.1f} menit)")
    print(f"  Best Val F1: {best_f1:.4f}")
    print(f"{'='*60}")

    # Load model terbaik dan simpan
    # Load model terbaik untuk evaluasi akhir
    if best_model_state is not None:
        model.load_state_dict(best_model_state)


    # ===================== EVALUASI AKHIR (pada TEST set) =====================
    print(f"\n{'='*60}")
    print(f"  EVALUASI AKHIR PADA TEST SET (Model Terbaik)")
    print(f"{'='*60}\n")

    test_loss, test_acc, test_f1, all_preds, all_labels = evaluate(
        model, test_loader, eval_criterion, device
    )

    print(f"Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Test F1-Score : {test_f1:.4f}")
    print(f"Test Loss     : {test_loss:.4f}")

    # Classification Report
    target_names = [DISPLAY_LABELS[name] for name in CLASS_NAMES]
    print(f"\nClassification Report:")
    print("-" * 60)
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print("-" * 60)
    header = "         " + "  ".join([f"{DISPLAY_LABELS[c]:>10}" for c in CLASS_NAMES])
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{DISPLAY_LABELS[CLASS_NAMES[i]]:>10} " + "  ".join([f"{v:10d}" for v in row])
        print(row_str)

    print(f"\n{'='*60}")
    print(f"  Selesai! Model siap digunakan.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
