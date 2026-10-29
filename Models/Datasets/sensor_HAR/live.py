import os
import time

import cv2
import numpy as np
import requests
import torch

from train_acc import AccNet

here = os.path.dirname(os.path.abspath(__file__))
ck = torch.load(os.path.join(here, "acc_model.pt"), map_location="cpu", weights_only=False)
model = AccNet()
model.load_state_dict(ck["model"])
model.eval()
mean, std, LABELS = ck["mean"], ck["std"], ck["labels"]
COLORS = [(80, 200, 80), (255, 170, 0), (0, 140, 255), (200, 80, 200), (0, 210, 210), (255, 90, 90)]
FONT = cv2.FONT_HERSHEY_SIMPLEX

base = input("Phyphox address (jaise http://192.168.1.7:8080): ").strip().rstrip("/")
if not base.startswith("http"):
    base = "http://" + base

T, X, Y, Z = [], [], [], []
since = 0.0


def poll():
    global since
    q = (f"{base}/get?acc_time={since}&accX={since}|acc_time"
         f"&accY={since}|acc_time&accZ={since}|acc_time")
    b = requests.get(q, timeout=2).json()["buffer"]
    t, x, y, z = (b[k]["buffer"] for k in ("acc_time", "accX", "accY", "accZ"))
    n = min(len(t), len(x), len(y), len(z))
    if n == 0:
        return
    T.extend(t[:n]); X.extend(x[:n]); Y.extend(y[:n]); Z.extend(z[:n])
    since = T[-1]
    cut = next((i for i, v in enumerate(T) if v >= T[-1] - 8), 0)
    del T[:cut], X[:cut], Y[:cut], Z[:cut]


def predict():
    if len(T) < 50 or T[-1] - T[0] < 2.6:
        return None
    tt = np.linspace(T[-1] - 2.56, T[-1], 128)   # 128 points = 2.56 sec, 50 Hz jaisa
    # UCI axes (x, y, z) <- phone axes (Y, X, Z), m/s2 ko g me badalte hain
    sig = np.stack([np.interp(tt, T, Y), np.interp(tt, T, X), np.interp(tt, T, Z)]) / 9.81
    sig = (sig - mean[0]) / std[0]
    x = torch.from_numpy(sig[None].astype(np.float32))
    with torch.no_grad():
        return torch.softmax(model(x), dim=1)[0].numpy()


def draw(label, probs, status):
    img = np.full((420, 640, 3), 25, np.uint8)
    k = int(np.argmax(probs)) if probs.sum() > 0 else None
    col = COLORS[k] if k is not None else (200, 200, 200)
    cv2.putText(img, label, (20, 70), FONT, 1.6, col, 4)
    cv2.putText(img, status, (20, 105), FONT, 0.6, (0, 255, 255), 1)
    for i, name in enumerate(LABELS):
        y0 = 140 + i * 44
        cv2.putText(img, name, (20, y0 + 22), FONT, 0.6, (255, 255, 255), 1)
        cv2.rectangle(img, (190, y0), (610, y0 + 28), (60, 60, 60), -1)
        cv2.rectangle(img, (190, y0), (190 + int(420 * probs[i]), y0 + 28), COLORS[i], -1)
        cv2.putText(img, f"{probs[i] * 100:.0f}%", (560, y0 + 21), FONT, 0.55, (255, 255, 255), 1)
    cv2.imshow("Sensor HAR Live (q = quit)", img)


probs, hist, label, status, last = np.zeros(6), [], "Collecting data...", "", 0.0
while True:
    try:
        poll()
        status = f"Phyphox connected | samples: {len(T)}"
    except Exception:
        status = "Phyphox se connect nahi ho raha (Play on hai? same WiFi?)"

    if time.time() - last >= 1.0:
        p = predict()
        if p is not None:
            hist = (hist + [p])[-3:]
            probs = np.mean(hist, axis=0)
            label = LABELS[int(np.argmax(probs))]
        last = time.time()

    draw(label, probs, status)
    if cv2.waitKey(150) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()