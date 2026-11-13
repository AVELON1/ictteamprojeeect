# Presentation Slides: Robust Road Sign Recognition (Option B)
**Topic:** Real-Time Traffic Sign Detection Under Adverse Environmental Conditions via OpenCV Synthetic Data Augmentation  
**Team Project:** ICT Assignment (Data-Centric AI Track)

---

## Slide 1: Title & Overview
- **Title:** Robust Vision for Autonomous Vehicles via Synthetic Data Augmentation
- **Approach:** Option B (Data-Centric AI)
- **Key Insight:** Improving real-world model reliability by enriching training data with OpenCV physical distortions rather than complicating the neural network architecture.

---

## Slide 2: The Practical Problem
- Autonomous vehicle cameras operate in uncontrolled real-world environments:
  - Motion blur caused by vehicle velocity
  - Severe angle inclination ($\pm 15^\circ$ to $25^\circ$)
  - Low-light night conditions and headlight overexposure
  - Adverse weather (fog, precipitation on windshield)
- **The Danger of "Naive" Models:**
  - A model trained purely on clean internet photos achieves ~99% on pristine benchmarks, but fails catastrophically under real distortions.
  - **Live Example:** Misidentifying a **Stop** sign as **No Entry** with 80.8% false confidence!

---

## Slide 3: Controlled Methodology (Option B)
- **Controlled Experiment:** Both models share the exact same CNN architecture (~320K parameters, 3 Conv blocks + Dense classifier).
- **Model 1 (Baseline):** Trained strictly on clean, pristine sign images.
- **Model 2 (Enriched):** Trained on clean data + real-time OpenCV synthetic copies:
  - `cv2.getRotationMatrix2D` + `cv2.warpAffine` ($\pm 15^\circ$ rotation)
  - `cv2.GaussianBlur` (optical defocus simulation)
  - `cv2.convertScaleAbs` (contrast & exposure adjustment)
  - *No horizontal flips* (preserves directional road sign semantics).

---

## Slide 4: Quantitative Results & Metrics
- **Clean Test Accuracy:**
  - Model 1: **99.26%** | Model 2: **99.88%**
- **Adverse / Distorted Test Accuracy:**
  - Model 1: **72.43%** (Sharp degradation, erratic predictions)
  - Model 2: **88.50% - 92.4%** (**+16% accuracy retention**)
- **Inference Speed:**
  - < 8 ms latency per image on standard CPU (zero runtime compute penalty).

---

## Slide 5: Live Interactive System (Streamlit Demo)
- Real-time dual-telemetry dashboard (`http://localhost:8501`):
  - **Unseen Edge Cases:** Pre-built severe scenarios (heavy fog, tilted camera, night dusk).
  - **Custom Input:** User photo upload and dashcam video frame analysis.
  - **Live Perturbation Sliders:** Move blur, angle, and darkness sliders in real-time to watch Model 1 fail while Model 2 stays resilient.

---

## Slide 6: Conclusion & Business Impact
1. Data quality and environmental variability in training data outweigh raw parameter count.
2. Synthetic enrichment via OpenCV is zero-cost, reproducible, and easily deployable to edge devices.
3. Full reproducible repository and live demo available on GitHub.
