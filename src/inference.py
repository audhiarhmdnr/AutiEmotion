"""
Skrip Inferensi Real-Time untuk Deteksi Emosi ASD (6 Kelas).
Menggunakan webcam + OpenCV Face Detection + Model yang sudah dilatih.

Fitur Stabilisasi:
- Exponential Moving Average (EMA) pada probabilitas prediksi
- Sliding window voting (5 frame terakhir)
- Confidence threshold minimum
- Test-Time Augmentation (TTA) ringan
"""

import os
import sys
import cv2
import time
import torch
import numpy as np
from torchvision import transforms
from collections import deque
from eeg_module import predict_stress, get_focus
from rulebase import get_recommendation

sys.path.insert(0, os.path.dirname(__file__))
from dataset import CLASS_NAMES, DISPLAY_LABELS, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD
from model import EmotionClassifier

# Warna untuk setiap kelas (BGR format)
EMOTION_COLORS = {
    'Natural':  (0, 200, 100),     # Hijau
    'anger':    (0, 0, 255),       # Merah
    'fear':     (0, 100, 255),     # Oranye
    'joy':      (0, 255, 255),     # Kuning
    'sadness':  (255, 100, 0),     # Biru
    'surprise': (255, 0, 200),     # Magenta
}

# Emoji sederhana untuk setiap kelas
EMOTION_EMOJI = {
    'Natural':  ':)',
    'anger':    '>:(',
    'fear':     'D:',
    'joy':      ':D',
    'sadness':  ':(',
    'surprise': ':O',
}

# ======================== STABILIZER ========================

class PredictionStabilizer:
    """
    Menstabilkan prediksi emosi menggunakan beberapa teknik:
    1. Exponential Moving Average (EMA) pada probabilitas
    2. Sliding window majority voting
    3. Confidence threshold
    """

    def __init__(self, num_classes=6, ema_alpha=0.6, window_size=7,
                 confidence_threshold=0.35, switch_threshold=0.5):
        self.num_classes = num_classes
        self.ema_alpha = ema_alpha
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.switch_threshold = switch_threshold

        # State
        self.ema_probs = np.ones(num_classes) / num_classes
        self.prediction_history = deque(maxlen=window_size)
        self.current_label = None
        self.frames_since_init = 0

    def update(self, raw_probs):
        """
        Update stabilizer dengan probabilitas prediksi baru.

        Returns:
            (class_name, confidence, smoothed_probs, is_stable)
        """
        self.frames_since_init += 1

        # 1. EMA smoothing pada probabilitas
        if self.frames_since_init == 1:
            self.ema_probs = raw_probs.copy()
        else:
            self.ema_probs = self.ema_alpha * raw_probs + (1 - self.ema_alpha) * self.ema_probs

        self.ema_probs = self.ema_probs / self.ema_probs.sum()

        # 2. Tentukan prediksi dari EMA
        ema_pred = np.argmax(self.ema_probs)
        ema_conf = self.ema_probs[ema_pred]

        # 3. Tambah ke history untuk majority voting
        self.prediction_history.append(ema_pred)

        # 4. Majority voting dari sliding window
        if len(self.prediction_history) >= 3:
            votes = np.bincount(list(self.prediction_history), minlength=self.num_classes)
            majority_class = np.argmax(votes)
            majority_ratio = votes[majority_class] / len(self.prediction_history)

            if self.current_label is None:
                self.current_label = majority_class
            elif majority_class != self.current_label:
                if majority_ratio >= self.switch_threshold:
                    self.current_label = majority_class
        else:
            self.current_label = ema_pred

        # 5. Confidence check
        final_class = self.current_label
        final_conf = self.ema_probs[final_class]
        is_stable = final_conf >= self.confidence_threshold

        class_name = CLASS_NAMES[final_class] if is_stable else None

        return class_name, final_conf, self.ema_probs, is_stable

    def reset(self):
        self.ema_probs = np.ones(self.num_classes) / self.num_classes
        self.prediction_history.clear()
        self.current_label = None
        self.frames_since_init = 0

# =============================================================


def load_model(model_path, device):
    """Load model yang sudah dilatih."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    num_classes = len(checkpoint.get('class_names', CLASS_NAMES))

    model = EmotionClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[INFO] Model berhasil dimuat dari: {model_path}")
    print(f"[INFO] Jumlah kelas: {num_classes} ({checkpoint.get('class_names', CLASS_NAMES)})")
    print(f"[INFO] Best F1 saat training: {checkpoint.get('best_f1', 'N/A')}")
    return model


def get_preprocess_transform():
    """Transform standar untuk preprocessing wajah."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_tta_transforms():
    """Test-Time Augmentation transforms — asli + flipped."""
    return [
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]),
    ]


def predict_emotion(model, face_img, device, tta_transforms=None):
    """
    Prediksi emosi dari gambar wajah (RGB).
    Dengan TTA: rata-ratakan probabilitas dari beberapa variasi.
    """
    with torch.no_grad():
        if tta_transforms:
            all_probs = []
            for transform in tta_transforms:
                tensor = transform(face_img).unsqueeze(0).to(device)
                output = model(tensor)
                probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                all_probs.append(probs)
            avg_probs = np.mean(all_probs, axis=0)
        else:
            transform = get_preprocess_transform()
            tensor = transform(face_img).unsqueeze(0).to(device)
            output = model(tensor)
            avg_probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    return avg_probs


def detect_faces_haar(frame):
    """Deteksi wajah menggunakan Haar Cascade."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return list(faces)


def draw_results(frame, bbox, class_name, confidence, all_probs, is_stable):
    """Gambar hasil prediksi di frame."""
    x, y, w, h = bbox

    if not is_stable or class_name is None:
        color = (128, 128, 128)
        label = "Uncertain (?)"
    else:
        color = EMOTION_COLORS.get(class_name, (255, 255, 255))
        display_name = DISPLAY_LABELS.get(class_name, class_name)
        emoji = EMOTION_EMOJI.get(class_name, '')
        label = f"{display_name} {emoji} ({confidence*100:.1f}%)"



    # Label background
    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w + 10, y), color, -1)
    cv2.putText(frame, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Bar chart untuk probabilitas
    bar_x = x + w + 10
    bar_y = y
    bar_width = 120
    bar_height = 18

    for i, (cls, prob) in enumerate(zip(CLASS_NAMES, all_probs)):
        cls_display = DISPLAY_LABELS[cls]
        cls_color = EMOTION_COLORS.get(cls, (200, 200, 200))

        cv2.rectangle(frame, (bar_x, bar_y + i * 24),
                      (bar_x + bar_width, bar_y + i * 24 + bar_height), (50, 50, 50), -1)
        filled_width = int(bar_width * prob)
        cv2.rectangle(frame, (bar_x, bar_y + i * 24),
                      (bar_x + filled_width, bar_y + i * 24 + bar_height), cls_color, -1)
        cv2.putText(frame, f"{cls_display} {prob*100:.0f}%",
                    (bar_x - 5, bar_y + i * 24 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame


def main():
    # Path model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'emotion_model_best.pth')

    if not os.path.exists(model_path):
        print(f"[ERROR] Model tidak ditemukan di: {model_path}")
        print("[INFO] Jalankan train.py terlebih dahulu untuk melatih model.")
        return

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")
    if device.type == 'cuda':
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    model = load_model(model_path, device)

    # TTA transforms
    tta_transforms = get_tta_transforms()
    use_tta = True
    print(f"[INFO] Test-Time Augmentation: {'ON' if use_tta else 'OFF'}")

    # Prediction Stabilizer
    stabilizer = PredictionStabilizer(
        num_classes=len(CLASS_NAMES),
        ema_alpha=0.6,
        window_size=7,
        confidence_threshold=0.35,
        switch_threshold=0.5
    )
    print("[INFO] Prediction Stabilizer aktif (EMA + Sliding Window + Confidence Threshold)")

    # Face Detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("[ERROR] Gagal memuat Haar Cascade classifier!")
        return
    print("[INFO] Face detector (Haar Cascade) berhasil dimuat.")

    # Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("[ERROR] Tidak bisa membuka webcam!")
        return

    prev_time = 0
    print(f"\n[INFO] Deteksi Emosi ASD Real-Time aktif!")
    print(f"[INFO] 6 Kelas: {', '.join([DISPLAY_LABELS[c] for c in CLASS_NAMES])}")
    print("[INFO] Tekan 'q' untuk keluar.")
    print("[INFO] Tekan 't' untuk toggle TTA on/off.\n")

    eeg_sample = {
        "Raw": 37,
        "Attention": 69,
        "Meditation": 35,
        "delta": 797033,
        "low-alpha": 220853,
        "high-alpha": 15504,
        "low-beta": 8871,
        "high-beta": 202271
    }

    stress = predict_stress(eeg_sample)
    focus = get_focus(eeg_sample["Attention"])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror
        h, w, _ = frame.shape

        # Deteksi wajah
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
        )

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for (x, y, fw, fh) in faces:
            x2 = min(w, x + fw)
            y2 = min(h, y + fh)

            face_crop = rgb_frame[y:y2, x:x2]

            if face_crop.size > 0:
                raw_probs = predict_emotion(
                    model, face_crop, device,
                    tta_transforms=tta_transforms if use_tta else None
                )

                class_name, confidence, smoothed_probs, is_stable = stabilizer.update(raw_probs)

                if class_name is not None:

                    emotion = class_name

                    terapi = get_recommendation(
                        emotion,
                        stress,
                        focus
                    )

                    if class_name is not None:
                        emotion = class_name

                        terapi = get_recommendation(
                            emotion,
                            stress,
                            focus
                        )

                        cv2.putText(
                            frame,
                            f"Stress: {stress}",
                            (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255,255,255),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"Focus: {focus}",
                            (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255,255,255),
                            2
                        )

                        cv2.putText(
                            frame,
                            f"ABA: {terapi}",
                            (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,255,0),
                            2
                        )

                frame = draw_results(
                    frame,
                    (x, y, fw, fh),
                    class_name,
                    confidence,
                    smoothed_probs,
                    is_stable
                )

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Status info
        tta_status = "TTA: ON" if use_tta else "TTA: OFF"
        cv2.putText(frame, tta_status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        # Judul
        cv2.putText(frame, 'ASD Emotion Detection (6 Classes - Stabilized)', (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('ASD Emotion Detection - Real Time', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            use_tta = not use_tta
            print(f"[INFO] TTA {'ON' if use_tta else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Program dihentikan.")


if __name__ == '__main__':
    main()
