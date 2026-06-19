import os
import cv2
import time
import torch
import numpy as np

from src.inference import (
    load_model,
    get_tta_transforms,
    predict_emotion,
    PredictionStabilizer,
    draw_results,
    CLASS_NAMES,
    DISPLAY_LABELS
)

from src.eeg_module import predict_stress, get_focus
from src.rules import get_rule

latest_result = {
    "emotion": "-",
    "confidence": 0,
    "stress": "-",
    "focus": "-",
    "therapy": "-"
}


def generate_frames():

    global latest_result

    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "emotion_model_best.pth"
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = load_model(model_path, device)

    tta_transforms = get_tta_transforms()

    stabilizer = PredictionStabilizer(
        num_classes=len(CLASS_NAMES),
        ema_alpha=0.6,
        window_size=7,
        confidence_threshold=0.35,
        switch_threshold=0.5
    )

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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

    prev_time = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        for (x, y, fw, fh) in faces:

            x2 = min(w, x + fw)
            y2 = min(h, y + fh)

            face_crop = rgb_frame[y:y2, x:x2]

            if face_crop.size == 0:
                continue

            raw_probs = predict_emotion(
                model,
                face_crop,
                device,
                tta_transforms
            )

            class_name, confidence, smoothed_probs, is_stable = (
                stabilizer.update(raw_probs)
            )

            therapy = "-"

            if class_name:

                rule = get_rule(class_name)
                therapy = rule["therapy"]
                avatar = rule["avatar"]
                print("EMOTION RAW:", class_name)
                print("AVATAR:", avatar)

                latest_result.update({
                    "emotion": class_name,
                    "confidence": round(confidence * 100, 1),
                    "stress": stress,
                    "focus": focus,
                    "therapy": therapy,
                    "avatar": avatar
                })

                cv2.putText(
                    frame,
                    f"Stress: {stress}",
                    (10,100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Focus: {focus}",
                    (10,130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2
                )

                cv2.putText(
                    frame,
                    f"ABA: {therapy}",
                    (10,160),
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

        curr_time = time.time()

        fps = (
            1/(curr_time-prev_time)
            if prev_time > 0
            else 0
        )

        prev_time = curr_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        ret, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame_bytes +
            b'\r\n'
        )