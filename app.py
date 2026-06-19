from flask import Flask, render_template, Response, jsonify
import cv2
import sys
import os
import signal
import threading

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.inference_web import generate_frames, latest_result

app = Flask(__name__)

# ── Graceful shutdown ──────────────────────────────────────────────────────────
_shutdown_event = threading.Event()

def _handle_sigint(sig, frame):
    print("\n[INFO] CTRL+C diterima — menghentikan server...")
    _shutdown_event.set()
    # Paksa exit agar Flask berhenti (termasuk thread reloader)
    os._exit(0)

signal.signal(signal.SIGINT, _handle_sigint)
signal.signal(signal.SIGTERM, _handle_sigint)
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route("/status")
def status():
    return jsonify(latest_result)

if __name__ == "__main__":
    print("[INFO] Server berjalan di http://127.0.0.1:5000")
    print("[INFO] Tekan CTRL+C untuk menghentikan server\n")
    app.run(debug=False, use_reloader=False, threaded=True)