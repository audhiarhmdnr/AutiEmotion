"""
Offline Data Augmentation untuk Dataset Emosi ASD.
Memperbanyak dataset training dengan membuat variasi augmented per gambar.
Menggunakan cv2 + numpy saja (tanpa albumentations) agar cepat.

Dari ~200 gambar/kelas → ~1200 gambar/kelas (asli + 5 variasi).
Output disimpan ke data/augmented/train/ dengan struktur folder yang sama.
"""

import os
import sys
import random
import shutil

sys.stdout.reconfigure(line_buffering=True)
print("Memulai augment_dataset.py...", flush=True)

print("  Loading cv2...", flush=True)
import cv2
print("  Loading numpy...", flush=True)
import numpy as np
print("  Library siap!", flush=True)

# Konfigurasi
NUM_AUGMENTATIONS = 5
IMG_SIZE = 224
ORIGINAL_FOLDERS = ['Natural', 'anger', 'fear', 'joy', 'sadness', 'surprise']


# ==================== AUGMENTASI DENGAN CV2 ====================

def random_flip(image):
    """Horizontal flip secara acak."""
    if random.random() > 0.5:
        return cv2.flip(image, 1)
    return image


def random_brightness_contrast(image, brightness_limit=0.3, contrast_limit=0.3):
    """Ubah brightness dan contrast secara acak."""
    alpha = 1.0 + random.uniform(-contrast_limit, contrast_limit)  # contrast
    beta = random.uniform(-brightness_limit, brightness_limit) * 255  # brightness
    result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return result


def random_rotation(image, max_angle=25):
    """Rotasi gambar secara acak."""
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    scale = random.uniform(0.85, 1.15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    result = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return result


def random_blur(image):
    """Blur secara acak."""
    ksize = random.choice([3, 5])
    if random.random() > 0.5:
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    else:
        return cv2.blur(image, (ksize, ksize))


def random_noise(image):
    """Tambah Gaussian noise."""
    noise = np.random.normal(0, random.uniform(5, 20), image.shape).astype(np.float32)
    result = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return result


def random_hsv_shift(image):
    """Ubah hue, saturation, value secara acak."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] += random.uniform(-10, 10)  # hue
    hsv[:, :, 1] *= random.uniform(0.8, 1.2)  # saturation
    hsv[:, :, 2] *= random.uniform(0.8, 1.2)  # value
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def random_crop_resize(image, target_size=IMG_SIZE):
    """Random crop lalu resize."""
    h, w = image.shape[:2]
    crop_ratio = random.uniform(0.75, 0.95)
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    y = random.randint(0, h - crop_h)
    x = random.randint(0, w - crop_w)
    cropped = image[y:y + crop_h, x:x + crop_w]
    return cv2.resize(cropped, (target_size, target_size))


def random_erasing(image, max_holes=3):
    """Random erasing / cutout."""
    result = image.copy()
    h, w = image.shape[:2]
    num_holes = random.randint(1, max_holes)
    for _ in range(num_holes):
        hole_h = random.randint(10, 30)
        hole_w = random.randint(10, 30)
        y = random.randint(0, max(0, h - hole_h))
        x = random.randint(0, max(0, w - hole_w))
        result[y:y + hole_h, x:x + hole_w] = np.random.randint(0, 255, (hole_h, hole_w, 3), dtype=np.uint8)
    return result


def apply_clahe(image):
    """CLAHE untuk meningkatkan kontras lokal."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# 5 pipeline augmentasi berbeda
def augment_pipeline_1(image):
    """Geometric: flip + rotation + crop."""
    img = random_flip(image)
    img = random_rotation(img, max_angle=25)
    if random.random() > 0.5:
        img = random_crop_resize(img)
    else:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def augment_pipeline_2(image):
    """Color: brightness + contrast + HSV."""
    img = random_brightness_contrast(image, 0.35, 0.35)
    img = random_hsv_shift(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def augment_pipeline_3(image):
    """Blur + noise + flip."""
    img = random_blur(image)
    img = random_noise(img)
    img = random_flip(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def augment_pipeline_4(image):
    """CLAHE + rotation + brightness."""
    img = apply_clahe(image)
    img = random_rotation(img, max_angle=15)
    img = random_brightness_contrast(img, 0.2, 0.2)
    img = random_flip(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def augment_pipeline_5(image):
    """Erasing + flip + brightness."""
    img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = random_erasing(img)
    img = random_flip(img)
    img = random_brightness_contrast(img, 0.25, 0.25)
    return img


PIPELINES = [
    augment_pipeline_1,
    augment_pipeline_2,
    augment_pipeline_3,
    augment_pipeline_4,
    augment_pipeline_5,
]


# ==================== MAIN ====================

def augment_dataset(data_root, output_root, num_augmentations=NUM_AUGMENTATIONS):
    """Membuat dataset augmented dari dataset asli."""
    train_dir = os.path.join(data_root, 'train')
    output_dir = os.path.join(output_root, 'train')

    if not os.path.isdir(train_dir):
        print(f"[ERROR] Folder train tidak ditemukan: {train_dir}", flush=True)
        return

    print("=" * 60, flush=True)
    print("  OFFLINE DATA AUGMENTATION", flush=True)
    print(f"  Input : {train_dir}", flush=True)
    print(f"  Output: {output_dir}", flush=True)
    print(f"  Variasi per gambar: {num_augmentations}", flush=True)
    print("=" * 60, flush=True)

    total_original = 0
    total_augmented = 0

    for folder_name in ORIGINAL_FOLDERS:
        class_dir = os.path.join(train_dir, folder_name)
        if not os.path.isdir(class_dir):
            print(f"[WARNING] Folder '{folder_name}' tidak ditemukan, skip.", flush=True)
            continue

        output_class_dir = os.path.join(output_dir, folder_name)
        os.makedirs(output_class_dir, exist_ok=True)

        image_files = [
            f for f in os.listdir(class_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ]

        if not image_files:
            print(f"[WARNING] Tidak ada gambar di folder '{folder_name}'.", flush=True)
            continue

        count = len(image_files)
        print(f"\n[INFO] Memproses '{folder_name}': {count} gambar asli", flush=True)
        total_original += count

        for idx, fname in enumerate(image_files):
            fpath = os.path.join(class_dir, fname)
            image = cv2.imread(fpath)
            if image is None:
                print(f"  [WARNING] Gagal baca: {fpath}", flush=True)
                continue

            base, ext = os.path.splitext(fname)

            # 1. Copy gambar asli (resize ke IMG_SIZE)
            original_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
            original_out = os.path.join(output_class_dir, f"{base}_orig{ext}")
            cv2.imwrite(original_out, original_resized)
            total_augmented += 1

            # 2. Buat augmentasi
            for i in range(num_augmentations):
                pipeline = PIPELINES[i % len(PIPELINES)]
                try:
                    aug_image = pipeline(image)
                    aug_out = os.path.join(output_class_dir, f"{base}_aug{i+1}{ext}")
                    cv2.imwrite(aug_out, aug_image)
                    total_augmented += 1
                except Exception as e:
                    print(f"  [WARNING] Augmentasi gagal {fname} pipeline {i}: {e}", flush=True)

            # Progress setiap 50 gambar
            if (idx + 1) % 50 == 0 or (idx + 1) == count:
                print(f"  [{idx+1}/{count}] selesai...", flush=True)

        class_total = len(os.listdir(output_class_dir))
        print(f"  -> {folder_name}: {count} asli -> {class_total} total", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  SELESAI!", flush=True)
    print(f"  Gambar asli   : {total_original}", flush=True)
    print(f"  Total output  : {total_augmented}", flush=True)
    print(f"  Rasio         : {total_augmented/max(total_original,1):.1f}x", flush=True)
    print(f"  Output folder : {output_dir}", flush=True)
    print(f"{'=' * 60}", flush=True)


def copy_test_data(data_root, output_root):
    """Copy data test tanpa augmentasi."""
    test_src = os.path.join(data_root, 'test')
    test_dst = os.path.join(output_root, 'test')

    if os.path.exists(test_dst):
        shutil.rmtree(test_dst)

    shutil.copytree(test_src, test_dst)
    print(f"[INFO] Data test di-copy ke: {test_dst}", flush=True)


if __name__ == '__main__':
    data_root = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'Autism emotion recogition dataset')
    output_root = os.path.join(os.path.dirname(__file__), '..', 'data', 'augmented')

    augment_dataset(data_root, output_root, num_augmentations=NUM_AUGMENTATIONS)
    copy_test_data(data_root, output_root)

    print("\n[INFO] Dataset augmented siap digunakan untuk training!", flush=True)
    print(f"[INFO] Gunakan path: data/augmented/", flush=True)
