import os
import json
import cv2
import numpy as np

def generate_hard_unseen():
    with open('dataset/classes.json', 'r') as f:
        classes = json.load(f)
        
    out_dir = 'test_unseen'
    os.makedirs(out_dir, exist_ok=True)
    
    # We take images from test_clean that models never saw during training,
    # and apply severe real-world harsh conditions:
    # 1. Rain / Heavy mist (strong blur + low contrast)
    # 2. Night headlights glare (extreme local brightness + dark background)
    # 3. Slanted angle (dashcam mounted at an angle 22-25 deg)
    # 4. Severe motion blur (directional blur)
    
    signs_map = {
        "2.5": "Stop_SevereAngle_MotionBlur",
        "2.1": "PriorityRoad_NightDusk_HeavyGrain",
        "3.1": "NoEntry_HeavyFog_Defocus",
        "1.22": "Pedestrian_Night_LowLight",
        "3.27": "NoStopping_Tilted_CameraBlur",
        "4.1.1": "Straight_RainyGlass_Distorted",
        "2.4": "GiveWay_HighGlare_Overexposed"
    }
    
    manifest = []
    
    for code, scenario_name in signs_map.items():
        folder = os.path.join('dataset/test_clean', code)
        if not os.path.exists(folder):
            continue
        files = sorted([f for f in os.listdir(folder) if f.endswith('.png')])
        if not files:
            continue
        # pick the last file to ensure it wasn't picked as first sample
        src_path = os.path.join(folder, files[-1])
        img = cv2.imread(src_path)
        img = cv2.resize(img, (128, 128))
        h, w = img.shape[:2]
        
        hard_img = img.copy()
        
        if "SevereAngle" in scenario_name:
            # 22 degree tilt + motion blur
            M = cv2.getRotationMatrix2D((w//2, h//2), 22, 1.0)
            hard_img = cv2.warpAffine(hard_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            # motion blur kernel
            kernel = np.zeros((9, 9))
            kernel[4, :] = 1.0 / 9
            hard_img = cv2.filter2D(hard_img, -1, kernel)
            
        elif "Night" in scenario_name:
            # Heavy darkness (35% brightness) + sensor grain
            hard_img = cv2.convertScaleAbs(hard_img, alpha=0.45, beta=-35)
            noise = np.random.normal(0, 18, hard_img.shape).astype(np.float32)
            hard_img = np.clip(hard_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
        elif "HeavyFog" in scenario_name:
            # Low contrast + strong defocus
            hard_img = cv2.GaussianBlur(hard_img, (11, 11), 0)
            hard_img = cv2.convertScaleAbs(hard_img, alpha=0.55, beta=40)
            
        elif "Tilted" in scenario_name:
            # -18 degree rotation + blur
            M = cv2.getRotationMatrix2D((w//2, h//2), -18, 1.0)
            hard_img = cv2.warpAffine(hard_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            hard_img = cv2.GaussianBlur(hard_img, (9, 9), 0)
            
        elif "RainyGlass" in scenario_name:
            # Blur + noise + slight angle
            M = cv2.getRotationMatrix2D((w//2, h//2), 12, 1.0)
            hard_img = cv2.warpAffine(hard_img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            hard_img = cv2.GaussianBlur(hard_img, (7, 7), 0)
            hard_img = cv2.convertScaleAbs(hard_img, alpha=0.7, beta=-15)
            
        elif "HighGlare" in scenario_name:
            # Overexposed glare wash out
            hard_img = cv2.convertScaleAbs(hard_img, alpha=1.4, beta=50)
            hard_img = cv2.GaussianBlur(hard_img, (5, 5), 0)

        out_fname = f"{scenario_name}_{code}.png"
        out_fpath = os.path.join(out_dir, out_fname)
        cv2.imwrite(out_fpath, hard_img)
        
        manifest.append({
            "code": code,
            "title": f"{code} ({classes[code]}) - {scenario_name.replace('_', ' ')}",
            "file": out_fname
        })
        
    with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print(f"Generated {len(manifest)} harsh unseen benchmark samples in {out_dir}")

if __name__ == '__main__':
    generate_hard_unseen()
