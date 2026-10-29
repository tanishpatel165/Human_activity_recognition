import os
import glob
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
hits = glob.glob(os.path.join(here, "**", "activity_labels.txt"), recursive=True)
if not hits:
    raise SystemExit("activity_labels.txt nahi mili. Zip ko SensorHAR me extract kiya?")
root = os.path.dirname(hits[0])

SIGNALS = ["body_acc_x", "body_acc_y", "body_acc_z",
           "body_gyro_x", "body_gyro_y", "body_gyro_z",
           "total_acc_x", "total_acc_y", "total_acc_z"]

labels = [l.split()[1] for l in open(os.path.join(root, "activity_labels.txt")).read().splitlines()]

def load_split(split):
    sig_dir = os.path.join(root, split, "Inertial Signals")
    X = np.stack([np.loadtxt(os.path.join(sig_dir, f"{s}_{split}.txt")) for s in SIGNALS], axis=1)
    y = np.loadtxt(os.path.join(root, split, f"y_{split}.txt")).astype(int) - 1
    return X.astype(np.float32), y

X_train, y_train = load_split("train")
X_test, y_test = load_split("test")

print("Activities:", labels)
print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_test: ", X_test.shape, " y_test: ", y_test.shape)
print("Har activity ke kitne windows (train):",
      {labels[i]: int((y_train == i).sum()) for i in range(6)})

np.savez(os.path.join(here, "har_data.npz"),
         X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
print("Saved: har_data.npz")