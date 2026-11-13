# 🛡️ Robust Road Sign Recognition via Synthetic Data Augmentation (Option B)

A computer vision research project demonstrating the power of **Data-Centric AI**: improving deep learning robustness in adverse environmental conditions using synthetic OpenCV perturbations without increasing model complexity.

---

## 📌 Problem Statement & Motivation
Standard deep convolutional neural networks trained on pristine, studio-quality images often suffer from catastrophic degradation when deployed on real-world vehicle cameras. Factors such as:
- Heavy motion blur and optical defocus
- Steep camera mount tilt angles ($\pm 15^\circ$ to $25^\circ$)
- Low-light dusk conditions and night headlight glare
- Weather artifacts (mist, rain on windshield)

cause naive models to produce high-confidence false predictions (e.g., misclassifying a **Stop** sign as a **No Entry** sign with 80.8% false confidence), which poses a critical safety hazard for autonomous navigation.

---

## 🔬 Methodology: Option B (Data Augmentation via OpenCV)
We adopted a controlled comparative methodology:
1. **Identical Model Architectures:** Both models use the exact same 3-block CNN topology (~320K parameters) with Adam optimization and sparse categorical cross-entropy.
2. **Model 1 (Clean Baseline):** Trained strictly on pristine, unobstructed road sign samples.
3. **Model 2 (OpenCV Synthetic Enriched):** Trained on clean samples + realistic synthetic adverse variations generated on-the-fly via OpenCV:
   - **Rotation:** `cv2.getRotationMatrix2D` + `cv2.warpAffine` ($\pm 15^\circ$)
   - **Optical Defocus:** `cv2.GaussianBlur` ($k \in [3, 9]$)
   - **Exposure Scaling:** `cv2.convertScaleAbs` ($\alpha \in [0.6, 1.3]$, $\beta \in [-35, 25]$)
   - **Sensor Grain:** Additive Gaussian noise
   - *(Note: Horizontal flip is strictly avoided to prevent invalidating directional traffic rules).*

---

## 📊 Benchmark Results

| Evaluation Scenario | Model 1 (Clean Baseline) | Model 2 (OpenCV Enriched) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Clean Test Set** | **99.26%** | **99.88%** | +0.62% |
| **Adverse / Distorted Test Set** | **72.43%** | **84.19% - 92.4%** | **+11.76% to +20.0%** |
| **Inference Latency (CPU)** | < 8 ms | < 8 ms | Zero overhead |

---

## 🚗 Target Classes (10 Standard CIS Signs)
- `1.22`: Pedestrian crossing (Пешеходный переход)
- `2.1`: Priority road (Главная дорога)
- `2.4`: Give way (Уступите дорогу)
- `2.5`: Stop (Движение без остановки запрещено)
- `3.1`: No entry (Въезд запрещен)
- `3.20`: No overtaking (Обгон запрещен)
- `3.27`: No stopping (Остановка запрещена)
- `4.1.1`: Go straight (Движение прямо)
- `5.5`: One-way road (Дорога с односторонним движением)
- `6.4`: Parking (Парковка)

---

## 🚀 Quickstart & Demo Application

### 1. Clone & Set Up
```bash
git clone https://github.com/AVELON1/ictteamprojeeect.git
cd ictteamprojeeect

# Activate virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install tensorflow opencv-python streamlit scikit-learn matplotlib
```

### 2. Launch Interactive Dashboard
```bash
./run_demo.sh
# or directly:
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to access:
- **Harsh Unseen Scenarios:** Test pre-configured edge cases (rain, fog, tilt, dusk).
- **Custom Uploads:** Upload your own photos or vehicle dashcam videos.
- **Real-Time Sliders:** Interactively adjust blur radius, tilt angle, darkness, and sensor noise to witness Model 1 fail while Model 2 remains robust.

---

## 📁 Repository Structure
```
├── app.py                      # Interactive Streamlit dual-model comparison dashboard
├── run_demo.sh                 # One-click execution script
├── SPEC.md                     # Engineering specification & roadmap
├── benchmark.py                # Comparative evaluation on clean vs adverse tests
├── train_baseline.py           # Training pipeline for Model 1 (Clean)
├── train_augmented.py          # Training pipeline for Model 2 (Option B: OpenCV)
├── src/
│   ├── model.py                # CNN neural network architecture
│   ├── augment.py              # OpenCV synthetic perturbation pipeline
│   └── data_loader.py          # Data ingestion and batching utilities
├── models/
│   ├── model_clean.keras       # Trained weights for Model 1
│   ├── model_aug.keras         # Trained weights for Model 2
│   └── classes_map.json        # Class index mapping dictionary
└── test_unseen/                # Unseen edge-case test samples
```
