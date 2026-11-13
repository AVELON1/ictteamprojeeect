# Defense English Glossary & Key Terminology
**Course / Track:** Data-Centric AI (Option B: OpenCV Synthetic Augmentation)

---

### 1. Core Technical Vocabulary

| Term | Simple Definition | How to use in presentation |
| :--- | :--- | :--- |
| **Data-Centric AI** | Focusing on engineering the quality and diversity of training data rather than just scaling neural network parameter counts. | *"We adopted a Data-Centric AI framework, keeping the neural architecture constant while enriching data distribution."* |
| **Overconfidence Hazard** | When a naive model predicts a completely wrong class with dangerously high confidence (e.g. 80.8% false confidence on STOP). | *"The baseline model demonstrates catastrophic overconfidence under adverse motion blur."* |
| **Affine Transformation** | Mathematical linear mapping using rotation matrix and displacement (`cv2.warpAffine`) preserving points and straight lines. | *"We simulated camera mounting inclination via affine transformations within ±15 degrees."* |
| **Optical Defocus** | Optical blur caused by out-of-focus lenses or fast vehicle speed, modeled via Gaussian convolutions (`cv2.GaussianBlur`). | *"Gaussian kernels simulate real-world motion blur and camera defocus."* |
| **Directional Semantics** | The physical meaning of road signs depending on direction (e.g., arrow directions, overtaking lane restrictions). | *"Horizontal flipping was strictly omitted to preserve directional road sign semantics."* |
| **Post-Training Quantization** | Converting FP32 32-bit floating point model weights down to lightweight representations (TFLite). | *"We quantized our models to TFLite format, reducing storage from 13 MB to 1.1 MB (12x compression) with <5 ms inference."* |
| **Invariance** | The ability of the vision pipeline to consistently recognize the sign invariant to noise, contrast, and tilt. | *"OpenCV synthetic enrichment forces the neural network to learn invariant geometric features."* |

---

### 2. High-Impact Elevator Pitch (30 Seconds)
> *"Our project demonstrates Option B: Robust Traffic Sign Vision. In real-world driving, cameras encounter blur, steep angles, and night glare. A model trained only on clean data dangerously misclassifies a STOP sign as NO ENTRY with 80.8% false confidence. By applying controlled physical augmentations using OpenCV, our enhanced model correctly identifies the sign with 75.4%+ confidence, boosts degraded accuracy by +16%, and deploys as an ultra-lightweight 1.1 MB TFLite model."*
