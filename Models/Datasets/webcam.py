import os
import cv2
import torch
import numpy as np
from collections import deque
from torchvision.models.video import mc3_18

MODEL_PATH = os.path.join("Models", "mc318-ufc101-split-1.pth")
CLASS_ROOT = r"c:\Users\Dell\Downloads\UCF101\UCF-101"
classes = sorted(os.listdir(CLASS_ROOT))

# Model load
ckpt = torch.load(MODEL_PATH, map_location="cpu")
model = mc3_18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 101)
state = {k.replace("backbone.", "", 1): v for k, v in ckpt["model_state_dict"].items()}
model.load_state_dict(state, strict=True)
model.eval()

mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

def preprocess(frame):
    f = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    f = cv2.resize(f, (171, 128))
    return f[8:120, 29:141]

def predict(buffer):
    clip = np.array(buffer, dtype=np.float32) / 255.0
    clip = (clip - mean) / std
    clip = torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(clip), dim=1)[0]
    p, i = torch.max(probs, dim=0)
    return classes[i.item()], p.item() * 100

cap = cv2.VideoCapture(0)
buffer = deque(maxlen=16)
label, conf = "Collecting frames...", 0.0
count = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break

    count += 1
    if count % 3 == 0:
        buffer.append(preprocess(frame))
    if len(buffer) == 16 and count % 24 == 0:
        label, conf = predict(buffer)

    color = (0, 200, 0) if conf >= 60 else (0, 165, 255)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (30, 30, 30), -1)
    cv2.putText(frame, f"{label}  {conf:.0f}%", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("HAR Live Detection (press q to quit)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Metrics:", checkpoint["metrics"])