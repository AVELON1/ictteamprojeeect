# Project Specification: Robust Road Sign Recognition via Synthetic Data Augmentation (Option B)

## 1. Direction & Spec
- **Objective**: Demonstrate that a baseline CNN model trained only on clean road sign images degrades sharply under adverse camera conditions (blur, camera tilt, darkness), whereas a model enriched with OpenCV synthetic augmentation maintains high accuracy (~92%+).
- **Target Classes** (10 CIS road sign categories from `dataset/classes.json`):
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
- **Model Architecture**:
  - Input: `(64, 64, 3)` normalized `[0, 1]`
  - Conv Block 1: `Conv2D(32, 3x3)` + `BatchNormalization` + `ReLU` + `MaxPooling2D(2x2)`
  - Conv Block 2: `Conv2D(64, 3x3)` + `BatchNormalization` + `ReLU` + `MaxPooling2D(2x2)`
  - Conv Block 3: `Conv2D(128, 3x3)` + `BatchNormalization` + `ReLU` + `MaxPooling2D(2x2)`
  - Classifier: `Flatten` + `Dense(128, relu)` + `Dropout(0.4)` + `Dense(10, softmax)`
- **OpenCV Augmentation Pipeline**:
  - Blur: `cv2.GaussianBlur` (kernel size 5 to 11)
  - Rotation: `cv2.getRotationMatrix2D` + `cv2.warpAffine` (angle $\pm 15^\circ$)
  - Lighting: Brightness & contrast scale ($0.4 \times$ to $1.3 \times$)
- **Artifacts**:
  - `models/model_clean.keras`: Trained on clean images only.
  - `models/model_aug.keras`: Trained on clean + augmented images.
  - `app.py`: Streamlit interactive comparison dashboard with real-time perturbation sliders.

## 2. Execution Checklist
- [x] Unzip and inspect dataset in `iiiiicct/dataset`
- [ ] Prepare isolated `.venv` with TensorFlow, OpenCV, Streamlit
- [ ] Implement data loader and OpenCV augmentation module (`src/augment.py`)
- [ ] Train Model 1 (Baseline / Clean) -> `models/model_clean.keras`
- [ ] Train Model 2 (Augmented / Robust) -> `models/model_aug.keras`
- [ ] Benchmark both models on clean test set vs adverse/distorted test set
- [ ] Build interactive Streamlit demo (`app.py`) with perturbation sliders
- [ ] Test and verify local app launch
