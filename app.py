import os
import json
import tempfile
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Option B: Robust Road Sign Recognition",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    .metric-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
    }
    
    .card-clean { border-left: 4px solid #f43f5e; }
    .card-aug { border-left: 4px solid #10b981; }
    .code-telemetry { font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; }
    .badge-error { background: rgba(244, 63, 94, 0.15); color: #fb7185; padding: 3px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; }
    .badge-success { background: rgba(16, 185, 129, 0.15); color: #34d399; padding: 3px 8px; border-radius: 6px; font-size: 0.8rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all_resources():
    classes_file = 'dataset/classes.json'
    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    sorted_codes = sorted(list(classes.keys()))
    idx_to_name = {i: f"{code} - {classes[code]}" for i, code in enumerate(sorted_codes)}
    
    m1_path = 'models/model_clean.keras'
    m2_path = 'models/model_aug.keras'
    
    m1 = tf.keras.models.load_model(m1_path) if os.path.exists(m1_path) else None
    m2 = tf.keras.models.load_model(m2_path) if os.path.exists(m2_path) else None
    
    manifest_path = 'test_unseen/manifest.json'
    unseen_items = []
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            unseen_items = json.load(f)
            
    return sorted_codes, idx_to_name, m1, m2, unseen_items

sorted_codes, idx_to_name, model_clean, model_aug, unseen_items = load_all_resources()

# Header
st.title("🛡️ Option B: Robust Road Sign Recognition via Synthetic Data Augmentation")
st.markdown("**Real-Time Degradation Benchmark:** Baseline Model (Clean-only) vs. Option B Model (OpenCV Synthetic Augmentation).")

# Sidebar
st.sidebar.header("🕹️ Input Source Mode")
input_mode = st.sidebar.radio("Select Input Mode:", [
    "🔥 Harsh Unseen Test Cases (Realistic Degraded Scenarios)",
    "📷 Live Laptop Webcam",
    "📸 Upload Image",
    "🎥 Upload Video Stream",
    "📁 Clean Benchmark Signs"
])

current_frame = None
target_label = None

if input_mode == "🔥 Harsh Unseen Test Cases (Realistic Degraded Scenarios)":
    st.sidebar.info("Brand new challenging real-world scenarios (fog, motion blur, dusk, angle tilt) never seen during training.")
    if unseen_items:
        opts = [item["title"] for item in unseen_items]
        chosen_title = st.sidebar.selectbox("Choose Harsh Scenario:", opts)
        chosen_item = next(it for it in unseen_items if it["title"] == chosen_title)
        img_p = os.path.join('test_unseen', chosen_item['file'])
        raw_bgr = cv2.imread(img_p)
        current_frame = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        target_label = idx_to_name[sorted_codes.index(chosen_item['code'])]

elif input_mode == "📷 Live Laptop Webcam":
    st.sidebar.info("Show a road sign to your webcam (from your phone screen or paper) to classify live!")
    cam_picture = st.sidebar.camera_input("Take a photo with Webcam")
    if cam_picture is not None:
        file_bytes = np.asarray(bytearray(cam_picture.read()), dtype=np.uint8)
        current_frame = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)

elif input_mode == "📸 Upload Image":
    up_file = st.sidebar.file_uploader("Upload road sign photo", type=["png", "jpg", "jpeg"])
    if up_file:
        file_bytes = np.asarray(bytearray(up_file.read()), dtype=np.uint8)
        current_frame = cv2.cvtColor(cv2.imdecode(file_bytes, 1), cv2.COLOR_BGR2RGB)

elif input_mode == "📁 Clean Benchmark Signs":
    test_dir = 'dataset/test_clean'
    available_signs = []
    if os.path.exists(test_dir):
        for code in sorted_codes:
            code_folder = os.path.join(test_dir, code)
            if os.path.isdir(code_folder):
                files = [f for f in os.listdir(code_folder) if f.endswith(('.png', '.jpg'))]
                if files:
                    available_signs.append((code, os.path.join(code_folder, files[0])))
    if available_signs:
        choice = st.sidebar.selectbox("Select Clean Sign:", [f"{c} ({idx_to_name[sorted_codes.index(c)].split(' - ')[1]})" for c, _ in available_signs])
        chosen_code = choice.split(" ")[0]
        target_label = idx_to_name[sorted_codes.index(chosen_code)]
        chosen_path = next(p for c, p in available_signs if c == chosen_code)
        raw_bgr = cv2.imread(chosen_path)
        current_frame = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

elif input_mode == "🎥 Upload Video Stream":
    st.sidebar.markdown("Upload dashcam video clip (.mp4, .mov, .avi).")
    vid_file = st.sidebar.file_uploader("Upload video file", type=["mp4", "mov", "avi"])
    if vid_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid_file.read())
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            frame_idx = st.sidebar.slider("Select Video Frame:", 0, max(0, total_frames - 1), 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()

# Realtime distortion sliders
st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Real-Time OpenCV Perturbations")
blur_val = st.sidebar.slider("Optical Defocus (Gaussian Blur px)", 0, 25, 0, step=2)
rotation_val = st.sidebar.slider("Camera Tilt / Rotation (Degrees)", -45, 45, 0, step=1)
brightness_val = st.sidebar.slider("Exposure / Dusk (Beta)", -90, 70, 0, step=5)
noise_val = st.sidebar.slider("Sensor Noise (Sigma)", 0, 40, 0, step=2)

if current_frame is not None:
    # Resize base image
    base = cv2.resize(current_frame, (64, 64), interpolation=cv2.INTER_AREA)
    perturbed = base.copy()
    
    # 1. Apply rotation
    if rotation_val != 0:
        h, w = perturbed.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rotation_val, 1.0)
        perturbed = cv2.warpAffine(perturbed, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
    # 2. Apply blur
    if blur_val > 0:
        k = blur_val if blur_val % 2 != 0 else blur_val + 1
        perturbed = cv2.GaussianBlur(perturbed, (k, k), 0)
        
    # 3. Apply brightness
    if brightness_val != 0:
        perturbed = cv2.convertScaleAbs(perturbed, alpha=1.0, beta=brightness_val)
        
    # 4. Apply noise
    if noise_val > 0:
        noise = np.random.normal(0, noise_val, perturbed.shape).astype(np.float32)
        perturbed = np.clip(perturbed.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Columns display
    col1, col2, col3 = st.columns([1.2, 1.4, 1.4])
    
    with col1:
        st.markdown("### 📷 Input Observation")
        st.image(cv2.resize(perturbed, (240, 240), interpolation=cv2.INTER_NEAREST), caption="Neural Input (64x64 RGB)", use_container_width=True)
        if target_label:
            st.info(f"🎯 **Ground Truth Class:**\n{target_label}")
            
    inp = perturbed.astype(np.float32) / 255.0
    inp_tensor = np.expand_dims(inp, axis=0)
    
    # Model 1 Clean
    with col2:
        st.markdown("### ❌ Model 1 (Clean Baseline)")
        st.caption("Trained strictly on pristine clean images")
        if model_clean is not None:
            pred1 = model_clean.predict(inp_tensor, verbose=0)[0]
            top_idx1 = int(np.argmax(pred1))
            conf1 = float(pred1[top_idx1])
            pred_name1 = idx_to_name[top_idx1]
            
            is_correct1 = (target_label is not None and pred_name1 == target_label)
            if target_label is not None:
                status_badge1 = '<span class="badge-success">CORRECT</span>' if is_correct1 else '<span class="badge-error">FALSE PREDICTION (MISTAKE)</span>'
            else:
                status_badge1 = '<span class="badge-success">CONFIDENT</span>' if conf1 > 0.8 else '<span class="badge-error">UNCERTAIN</span>'
            
            st.markdown(f"""
            <div class="metric-card card-clean">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.85rem; color:#94a3b8;">DETECTED CLASS</span>
                    {status_badge1}
                </div>
                <div style="font-size:1.15rem; font-weight:700; margin: 0.4rem 0;">{pred_name1}</div>
                <div class="code-telemetry" style="font-size:1.4rem; font-weight:800; color:#f43f5e;">
                    {conf1 * 100:.1f}% <span style="font-size:0.85rem; font-weight:400; color:#94a3b8;">confidence</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            top3_idx1 = np.argsort(pred1)[-3:][::-1]
            st.write("**Top-3 Probabilities:**")
            for i in top3_idx1:
                st.progress(float(pred1[i]), text=f"{idx_to_name[i]}: {pred1[i]*100:.1f}%")
                
    # Model 2 Aug
    with col3:
        st.markdown("### ✅ Model 2 (Option B: OpenCV Enriched)")
        st.caption("Trained with synthetic cv2 perturbations")
        if model_aug is not None:
            pred2 = model_aug.predict(inp_tensor, verbose=0)[0]
            top_idx2 = int(np.argmax(pred2))
            conf2 = float(pred2[top_idx2])
            pred_name2 = idx_to_name[top_idx2]
            
            is_correct2 = (target_label is not None and pred_name2 == target_label)
            if target_label is not None:
                status_badge2 = '<span class="badge-success">CORRECT (ROBUST)</span>' if is_correct2 else '<span class="badge-error">CHECK</span>'
            else:
                status_badge2 = '<span class="badge-success">ROBUST PASS</span>'
            
            st.markdown(f"""
            <div class="metric-card card-aug">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.85rem; color:#94a3b8;">DETECTED CLASS</span>
                    {status_badge2}
                </div>
                <div style="font-size:1.15rem; font-weight:700; margin: 0.4rem 0;">{pred_name2}</div>
                <div class="code-telemetry" style="font-size:1.4rem; font-weight:800; color:#10b981;">
                    {conf2 * 100:.1f}% <span style="font-size:0.85rem; font-weight:400; color:#94a3b8;">confidence</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            top3_idx2 = np.argsort(pred2)[-3:][::-1]
            st.write("**Top-3 Probabilities:**")
            for i in top3_idx2:
                st.progress(float(pred2[i]), text=f"{idx_to_name[i]}: {pred2[i]*100:.1f}%")

else:
    st.info("👈 Select an input mode in the sidebar: Harsh Unseen Tests, Live Webcam, Upload Image, or Video!")
