# 🧠 AutiEmotion — Sistem Deteksi Emosi untuk Anak ASD

> Sistem real-time berbasis web untuk mendeteksi emosi wajah dan tingkat stres pada anak dengan Autism Spectrum Disorder (ASD), disertai rekomendasi terapi ABA otomatis.

---

## 📋 Deskripsi Proyek

**AutiEmotion** adalah aplikasi web yang memanfaatkan *deep learning* dan *machine learning* untuk membantu pendidik, terapis, dan orang tua dalam memantau kondisi emosional anak ASD secara real-time. Sistem ini menggabungkan:

- 🎥 **Deteksi emosi wajah** via kamera menggunakan model EfficientNet-B0
- 🧬 **Prediksi tingkat stres** berbasis sinyal EEG menggunakan model klasik (`.pkl`)
- 📊 **Rekomendasi terapi ABA** (Applied Behavior Analysis) otomatis berdasarkan emosi yang terdeteksi
- 🌐 **Dashboard web interaktif** yang menampilkan hasil secara real-time

---

## 🗂️ Struktur Proyek

```
deteksi emosi ASD/
├── app.py                  # Entry point Flask server
├── requirements.txt        # Daftar dependensi Python
├── models/
│   ├── emotion_model_best.pth      # Model PyTorch (EfficientNet-B0)
│   └── model_stress_terbaik.pkl    # Model klasifikasi stres (EEG)
├── src/
│   ├── model.py            # Arsitektur EmotionClassifier
│   ├── inference.py        # Pipeline inferensi emosi
│   ├── inference_web.py    # Generator frame untuk streaming Flask
│   ├── eeg_module.py       # Modul prediksi stres dari data EEG
│   ├── rules.py            # Rule-based engine rekomendasi terapi ABA
│   ├── dataset.py          # Loader & preprocessing dataset
│   ├── augment_dataset.py  # Augmentasi data pelatihan
│   ├── train.py            # Script pelatihan model
│   ├── shared_state.py     # State global antar modul
│   └── rulebase.py         # Komponen rule base tambahan
├── templates/
│   └── dashboard.html      # Halaman utama dashboard web
└── static/
    ├── css/                # Stylesheet
    ├── js/                 # Script frontend
    └── avatars/            # Gambar avatar per emosi
```

---

## 🧩 Fitur Utama

| Fitur | Keterangan |
|---|---|
| 🎭 Deteksi Emosi | 6 kelas: `joy`, `sadness`, `anger`, `natural`, `surprise`, `fear` |
| 🧬 Prediksi Stres | Berbasis data EEG (Attention, Meditation, band frekuensi) |
| 🎯 Level Fokus | Dihitung dari nilai Attention EEG (≥60 = tinggi) |
| 💊 Rekomendasi ABA | Terapi disesuaikan otomatis berdasarkan emosi |
| 📹 Live Streaming | Feed video real-time via MJPEG di browser |
| 🖥️ Dashboard Web | Antarmuka modern berbasis HTML/CSS/JS |

---

## 🏗️ Arsitektur Model

### Deteksi Emosi (Vision)
- **Backbone**: EfficientNet-B0 (pre-trained ImageNet)
- **Classifier Head**: `Dropout → Linear(1280→512) → ReLU → BN → Dropout → Linear(512→256) → ReLU → BN → Dropout → Linear(256→6)`
- **Teknik Inferensi**: TTA (Test Time Augmentation) + EMA Smoothing + Prediction Stabilizer
- **Input**: Frame wajah dari webcam (224×224 px)

### Prediksi Stres (EEG)
- **Model**: Scikit-learn classifier (`.pkl`)
- **Fitur EEG**: `Raw`, `Attention`, `Meditation`, `delta`, `low-alpha`, `high-alpha`, `low-beta`, `high-beta`
- **Output**: `tinggi` / `rendah`

---

## ⚙️ Instalasi & Menjalankan

### 1. Clone Repositori

```bash
git clone https://github.com/audhiarhmdnr/AutiEmotion.git
cd AutiEmotion
```

### 2. Buat Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# atau
source .venv/bin/activate     # Linux/macOS
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
pip install flask joblib       # tambahan untuk web server & EEG model
```

> **Catatan**: Untuk GPU support, install PyTorch dengan CUDA sesuai versi driver Anda dari [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Pastikan Model Tersedia

Pastikan file model berikut ada di folder `models/`:
- `models/emotion_model_best.pth`
- `models/model_stress_terbaik.pkl`

### 5. Jalankan Aplikasi

```bash
python app.py
```

Buka browser dan akses: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

Tekan `CTRL+C` untuk menghentikan server.

---

## 🔌 Endpoint API

| Endpoint | Method | Keterangan |
|---|---|---|
| `/` | GET | Halaman dashboard utama |
| `/video_feed` | GET | MJPEG stream video dari kamera |
| `/status` | GET | JSON hasil deteksi terkini |

### Contoh Response `/status`

```json
{
  "emotion": "joy",
  "confidence": 87.3,
  "stress": "rendah",
  "focus": "tinggi",
  "therapy": "Social Play & Interaction",
  "avatar": "happy.png"
}
```

---

## 🧠 Rekomendasi Terapi ABA

Sistem secara otomatis memetakan emosi ke rekomendasi terapi berikut:

| Emosi | Rekomendasi Terapi |
|---|---|
| 😊 Joy | Social Play & Interaction |
| 😢 Sadness | Calming Music & Guided Breathing |
| 😠 Anger | Relaxation Exercises |
| 😐 Natural | Standard ABA Activities |
| 😲 Surprise | Exploratory Play |
| 😨 Fear | Calming Safety Exercises |

---

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **Deep Learning**: PyTorch, TorchVision (EfficientNet-B0)
- **Machine Learning**: Scikit-learn, Joblib
- **Computer Vision**: OpenCV
- **Data Processing**: NumPy, Pandas
- **Frontend**: HTML5, CSS3, JavaScript

---

## 📦 Requirements

```
opencv-python
torch
torchvision
numpy
matplotlib
scikit-learn
pandas
tqdm
flask
joblib
```

---

## 🚀 Pengembangan & Pelatihan Model

Untuk melatih ulang model emosi:

```bash
python src/train.py
```

Untuk augmentasi dataset:

```bash
python src/augment_dataset.py
```

---

## 👤 Author

**Audhi Arhmdn** — [@audhiarhmdnr](https://github.com/audhiarhmdnr)

Proyek ini dibuat sebagai bagian dari tugas akhir Semester 6 — Sistem Deteksi Emosi untuk Anak Autism Spectrum Disorder (ASD).

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik. Lisensi penggunaan mengikuti kebijakan institusi terkait.
