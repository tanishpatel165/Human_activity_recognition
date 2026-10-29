# 🏃 Human Activity Recognition (Video + Sensor)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Status](https://img.shields.io/badge/Status-Working-brightgreen)

Two deep learning pipelines that recognise what a person is doing, one from **video** and one from a **phone's accelerometer**.

## ✨ Results

| Pipeline | Model | Dataset | Classes | Accuracy |
|---|---|---|---|---|
| 🎥 Video | MC3-18 (3D CNN) | UCF101 | 101 | 87.05% (val) |
| 📱 Sensor | 1D-CNN | UCI HAR | 6 | 92.53% (test) |

## 🎥 1. Video-based HAR
- Loads the MC3-18 checkpoint and predicts the activity of a video clip (16 frames)
- Example: a Biking video was predicted as Biking with 91.1% confidence
- Live webcam inference: `Models/Datasets/webcam.py`

## 📱 2. Sensor-based HAR (real time)
- 1D-CNN trained on phone accelerometer data (3 channels)
- Activities: walking, upstairs, downstairs, sitting, standing, laying
- The phone streams live data through **Phyphox** (remote access) to the laptop,
  and `live.py` shows the predicted activity with confidence bars

## 🚀 Setup
    pip install -r requirements.txt

## ▶️ Run the sensor demo
1. Download the UCI HAR dataset into `Models/Datasets/sensor_HAR/`
2. Run `load_data.py`, then `train_acc.py` (creates `acc_model.pt`)
3. On the phone: Phyphox, Acceleration with g, enable remote access, press play
4. Run `live.py` and enter the Phyphox address (e.g. http://192.168.1.7:8080)

## 📁 Structure
- `Models/Datasets/main.py` : checkpoint verification
- `Models/Datasets/webcam.py` : live webcam prediction
- `Models/Datasets/sensor_HAR/` : sensor pipeline (load, train, live)

## ⚠️ Note
The large UCF101 checkpoint and the datasets are not stored in this repo.