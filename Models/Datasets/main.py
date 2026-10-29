import os
import torch

model_path = os.path.join("Models", "mc318-ufc101-split-1.pth")

print("HAR Project: Model Verification")
print("-" * 35)

if not os.path.isfile(model_path):
    print("Model file nahi mili!")
    print("File ko Models folder mein check karo.")
else:
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model mil gayi: {size_mb:.2f} MB")


checkpoint = torch.load(model_path, map_location="cpu")
print("Type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("Keys:", list(checkpoint.keys())[:10])


print("Epoch:", checkpoint["epoch"])
print("Best val acc:", checkpoint["best_val_acc"])
print("Config:", checkpoint["config"])    


from torchvision.models.video import mc3_18

model = mc3_18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 101)

state = checkpoint["model_state_dict"]
state = {k.replace("backbone.", "", 1): v for k, v in state.items()}
print(model.load_state_dict(state, strict=True))


model.eval()
print("Model load ho gaya")


import cv2
import numpy as np

VIDEO_PATH = r"c:\Users\Dell\Downloads\UCF101\UCF-101\Biking\v_Biking_g25_c04.avi"

def load_clip(path, num_frames=16):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    idx = np.linspace(0, len(frames) - 1, num_frames).astype(int)
    clip = []
    for i in idx:
        f = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        f = cv2.resize(f, (171, 128))
        f = f[8:120, 29:141]
        clip.append(f)

    clip = np.array(clip, dtype=np.float32) / 255.0
    mean = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
    std = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)
    clip = (clip - mean) / std
    return torch.from_numpy(clip).permute(3, 0, 1, 2).unsqueeze(0)

clip = load_clip(VIDEO_PATH)
with torch.no_grad():
    probs = torch.softmax(model(clip), dim=1)[0]

CLASS_ROOT = r"c:\Users\Dell\Downloads\UCF101\UCF-101"
classes = sorted(os.listdir(CLASS_ROOT))

top5 = torch.topk(probs, 5)
for p, i in zip(top5.values, top5.indices):
    print(f"{classes[i.item()]}: {p.item() * 100:.1f}%")
