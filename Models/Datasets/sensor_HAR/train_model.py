import os
import numpy as np
import torch
import torch.nn as nn

here = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(here, "har_data.npz"))
X_train, y_train, X_test, y_test = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
LABELS = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

# Normalize (har sensor channel ko same scale pe laate hain)
mean = X_train.mean(axis=(0, 2), keepdims=True)
std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

Xtr, ytr = torch.from_numpy(X_train), torch.from_numpy(y_train).long()
Xte, yte = torch.from_numpy(X_test), torch.from_numpy(y_test).long()


class HARNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(9, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 6))

    def forward(self, x):
        return self.fc(self.features(x).squeeze(-1))


if __name__ == "__main__":
    model = HARNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0

    for epoch in range(1, 21):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean().item() * 100
        print(f"Epoch {epoch:2d}/20   Test accuracy: {acc:.2f}%")

        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                        "labels": LABELS}, os.path.join(here, "har_sensor_model.pt"))

    print(f"\nBest accuracy: {best:.2f}%  (model saved: har_sensor_model.pt)")