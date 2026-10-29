import os
import pickle
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

DATA_FILE = "pose_data.pkl"
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = [(0, 200, 0), (255, 150, 0), (0, 165, 255), (200, 0, 200), (0, 200, 200),
          (255, 80, 80), (120, 200, 0), (0, 120, 255), (200, 200, 0)]
LIMBS = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
LIMB_COLORS = [(0, 255, 255), (0, 255, 0), (0, 200, 120), (255, 200, 0), (255, 140, 0),
               (255, 0, 200), (200, 0, 255), (255, 255, 0), (0, 150, 255), (0, 100, 255),
               (120, 255, 0), (80, 255, 80), (255, 255, 255), (255, 255, 255),
               (200, 200, 200), (200, 200, 200)]
WINDOW = 30    # kitne frames ki harkat dekhni hai
TARGET = 20    # ek baar me kitne samples sikhane hain

pose_model = YOLO("yolov8n-pose.pt")  # pehli baar auto-download hoga (internet chahiye)

if os.path.exists(DATA_FILE):
    with open(DATA_FILE, "rb") as f:
        data = pickle.load(f)
    labels = list(data.keys())
else:
    names = input("Gestures ke naam, comma se alag (max 9, English letters): ")
    labels = [n.strip() for n in names.split(",") if n.strip()][:9]
    data = {n: [] for n in labels}


def get_pose(frame):
    r = pose_model(frame, verbose=False)[0]
    if r.keypoints is None or len(r.keypoints.xy) == 0:
        return None
    return r.keypoints.xy[0].cpu().numpy()


def normalize(xy):
    if (xy[[5, 6, 11, 12]] == 0).any():
        return None
    c_sh = (xy[5] + xy[6]) / 2
    c_hip = (xy[11] + xy[12]) / 2
    torso = np.linalg.norm(c_sh - c_hip)
    if torso < 1:
        return None
    return ((xy - c_hip) / torso).flatten()


def feature(win):
    a = np.array(win)
    return np.concatenate([a.mean(0), a.std(0) * 2])


def predict(f):
    items = [(n, v) for n, vs in data.items() for v in vs]
    if len({n for n, _ in items}) < 2:
        return None, 0.0
    d = np.array([np.linalg.norm(v - f) for _, v in items])
    votes = {}
    for i in np.argsort(d)[:5]:
        votes[items[i][0]] = votes.get(items[i][0], 0) + 1 / (d[i] + 1e-3)
    best = max(votes, key=votes.get)
    return best, votes[best] / sum(votes.values()) * 100


def draw_skeleton(frame, xy):
    for (a, b), col in zip(LIMBS, LIMB_COLORS):
        if xy[a].min() > 0 and xy[b].min() > 0:
            cv2.line(frame, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), col, 3)
    for p in xy:
        if p.min() > 0:
            cv2.circle(frame, tuple(p.astype(int)), 5, (255, 255, 255), -1)


cap = cv2.VideoCapture(0)
win = deque(maxlen=WINDOW)
count = 0
label, conf = "Press 1-9 to teach me!", 0.0
mode, mode_label, mode_start, taken = None, None, 0, 0
msg = ""

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    count += 1
    now = time.time()

    xy = get_pose(frame)
    if xy is not None:
        draw_skeleton(frame, xy)
        n = normalize(xy)
        if n is not None:
            win.append(n)
    else:
        cv2.putText(frame, "Body nahi dikh rahi - peeche hato", (10, h - 20),
                    FONT, 0.7, (0, 0, 255), 2)

    if mode == "countdown":
        left = 3 - int(now - mode_start)
        if left <= 0:
            mode, taken = "record", 0
        else:
            cv2.putText(frame, str(left), (w // 2 - 50, h // 2 + 60),
                        FONT, 6, (0, 255, 255), 12)
    elif mode == "record":
        if count % 6 == 0 and len(win) == WINDOW:
            data[mode_label].append(feature(win))
            taken += 1
        col = COLORS[labels.index(mode_label) % len(COLORS)]
        cv2.rectangle(frame, (20, h - 50), (w - 20, h - 20), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, h - 50),
                      (20 + int((w - 40) * taken / TARGET), h - 20), col, -1)
        cv2.putText(frame, f"Recording {mode_label}... keep moving!", (20, h - 60),
                    FONT, 0.7, col, 2)
        if taken >= TARGET:
            mode = None
            msg = f"{mode_label} seekh liya! Total: {len(data[mode_label])}"
    elif len(win) == WINDOW and count % 10 == 0:
        name, c = predict(feature(win))
        if name:
            label, conf = name, c

    color = COLORS[labels.index(label) % len(COLORS)] if label in labels else (200, 200, 200)
    cv2.rectangle(frame, (0, 0), (w, 90), (30, 30, 30), -1)
    text = f"{label}  {conf:.0f}%" if conf else label
    cv2.putText(frame, text, (10, 35), FONT, 1, color, 2)
    counts = "  ".join(f"{i + 1}:{n}({len(data[n])})" for i, n in enumerate(labels))
    cv2.putText(frame, counts, (10, 62), FONT, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, msg, (10, 82), FONT, 0.5, (0, 255, 255), 1)
    cv2.imshow("Pose HAR (1-9 = teach, q = save and quit)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    idx = key - ord("1")
    if mode is None and 0 <= idx < len(labels) and len(win) == WINDOW:
        mode, mode_label, mode_start = "countdown", labels[idx], now
        msg = ""

cap.release()
cv2.destroyAllWindows()
with open(DATA_FILE, "wb") as f:
    pickle.dump(data, f)
print("Saved:", {n: len(v) for n, v in data.items()})