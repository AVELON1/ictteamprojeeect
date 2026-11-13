# Project Specification: Robust Road Sign Recognition via Synthetic Data Augmentation (Option B)

## 1. Direction & Spec
- **Objective**: Upgrade the interactive demonstration dashboard (`app.py`). Rather than requiring the user to manually pick an image and comparing only a single image, the landing experience immediately evaluates and displays all 10 CIS road sign categories simultaneously, presenting instant comparative statistics between Model 1 (fragile baseline without CV augmentation) and Model 2 (robust model with OpenCV synthetic perturbations).
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
- **Comparative Data Models**:
  - `models/model_clean.keras`: Baseline CNN trained strictly on clean images. Drops sharply under optical blur, rotation, and lighting changes (92.5% -> 64.3%).
  - `models/model_aug.keras`: Enhanced CNN trained with OpenCV synthetic perturbations (Gaussian blur, warpAffine rotation, contrast/brightness adjustments, sensor noise). Retains high robustness (99.8% -> 84.6%+).
- **Design Tokens & Aesthetic Architecture**:
  - Palette: Deep Slate theme (`#090d16` canvas, `#0f172a` container backgrounds, `#1e293b` borders, `#334155` dividers).
  - Status Accents: Rose (`#f43f5e`) for Model 1 / failures; Emerald (`#10b981`) for Model 2 / robust passes; Amber (`#f59e0b`) for moderate confidence.
  - Typography: `Plus Jakarta Sans` for headers and UI prose; `JetBrains Mono` for tabular metrics and telemetry.
  - Transitions: `<180ms ease-out`, no generic purple/indigo gradients or neon glow.

## 2. Multi-Phase Execution Checklist
- [x] **Milestone 1: Direction & Spec**
  - Updated SPEC.md with architecture, design tokens, data schema, and multi-category benchmarking flow.
- [ ] **Milestone 2: Core Implementation**
  - Create multi-category evaluation backend logic (`src/batch_eval.py` or within `app.py`).
  - Preload representative clean and degraded samples for all 10 categories.
  - Implement batched inference pipeline for both Model 1 and Model 2 with real-time accuracy and confidence calculations.
  - Test batch evaluation logic via CLI.
- [ ] **Milestone 3: Visual & Polish**
  - Design immediate multi-category landing view with simulated/live loading sequence on first visit.
  - Build executive statistical KPI cards (Accuracy, Error Rate Drop, Average Confidence, Resilience Factor).
  - Render comparative per-category interactive charts (Altair / Streamlit).
  - Construct responsive 10-category visual grid showing side-by-side inputs, predictions, and rescue tags.
  - Provide quick-preset degradation buttons (Weather, Optical Defocus, Tilt, Dusk) and real-time sliders.
  - Preserve secondary deep-dive modes (Single Sign Inspector, Webcam, Harsh Unseen Tests, Custom Upload) in structured tabs.
- [ ] **Milestone 4: Quality Gate**
  - Run the local Streamlit server on port 8501.
  - Verify auto-loading across all 10 categories upon page entry without manual intervention.
  - Audit against anti-slop rules (typography, palette, tabular numerals, performance).
