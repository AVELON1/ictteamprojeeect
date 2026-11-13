import os
import json
import cv2
import numpy as np
import tensorflow as tf
from src.data_loader import load_classes, load_dataset
from src.augment import apply_blur, apply_rotation, apply_brightness_contrast, apply_noise

CLASS_NAMES_RU = {
    "1.22": "Пешеходный переход",
    "2.1": "Главная дорога",
    "2.4": "Уступите дорогу",
    "2.5": "Движение без остановки запрещено (STOP)",
    "3.1": "Въезд запрещен (Кирпич)",
    "3.20": "Обгон запрещен",
    "3.27": "Остановка запрещена",
    "4.1.1": "Движение прямо",
    "5.5": "Дорога с односторонним движением",
    "6.4": "Парковка"
}

PRESET_PERFORM = {
    "mild_adverse": {
        "title": "Дорожные искажения (Легкие: вибрация, наклон 8°, легкий смаз)",
        "rotation": 8,
        "blur": 3,
        "brightness": -15,
        "noise": 8
    },
    "harsh_weather": {
        "title": "Тяжелые условия (Дождь/смаз + наклон 14° + сумерки + шум)",
        "rotation": 14,
        "blur": 7,
        "brightness": -35,
        "noise": 16
    },
    "heavy_fog": {
        "title": "Густой туман и оптический расфокус (Сильный Blur)",
        "rotation": 4,
        "blur": 11,
        "brightness": 20,
        "noise": 12
    },
    "night_dusk": {
        "title": "Ночная съемка / Грязная оптика (Низкая освещенность + зернистость)",
        "rotation": 6,
        "blur": 5,
        "brightness": -65,
        "noise": 22
    },
    "clean": {
        "title": "Идеальные условия (Лабораторные чистые снимки)",
        "rotation": 0,
        "blur": 0,
        "brightness": 0,
        "noise": 0
    }
}

def perturb_image(img, rotation=0, blur=0, brightness=0, noise=0):
    """Applies combined OpenCV perturbations."""
    out = img.copy()
    if rotation != 0:
        out = apply_rotation(out, rotation)
    if blur > 0:
        out = apply_blur(out, blur)
    if brightness != 0:
        out = apply_brightness_contrast(out, alpha=1.0, beta=brightness)
    if noise > 0:
        out = apply_noise(out, noise)
    return out

class ModelBenchmarkSuite:
    def __init__(self, data_dir='dataset/test_clean', classes_path='dataset/classes.json',
                 m1_path='models/model_clean.keras', m2_path='models/model_aug.keras'):
        self.codes, self.code_to_idx, self.idx_to_name = load_classes(classes_path)
        self.X_test, self.y_test = load_dataset(data_dir, self.code_to_idx)
        
        self.model_clean = tf.keras.models.load_model(m1_path) if os.path.exists(m1_path) else None
        self.model_aug = tf.keras.models.load_model(m2_path) if os.path.exists(m2_path) else None
        
        # Pick 1 representative image per category for instant visual comparison
        self.sample_per_category = {}
        for code, idx in self.code_to_idx.items():
            matches = np.where(self.y_test == idx)[0]
            if len(matches) > 0:
                # pick a sharp, representative middle sample
                pick_idx = matches[min(len(matches) // 2, len(matches) - 1)]
                self.sample_per_category[code] = {
                    'idx': idx,
                    'code': code,
                    'ru_name': CLASS_NAMES_RU.get(code, self.idx_to_name[idx]),
                    'img_rgb': self.X_test[pick_idx]
                }

    def evaluate_preset(self, preset_key="harsh_weather"):
        cfg = PRESET_PERFORM.get(preset_key, PRESET_PERFORM["harsh_weather"])
        return self.evaluate_custom(
            rotation=cfg["rotation"],
            blur=cfg["blur"],
            brightness=cfg["brightness"],
            noise=cfg["noise"],
            preset_title=cfg["title"]
        )

    def evaluate_custom(self, rotation=0, blur=0, brightness=0, noise=0, preset_title="Пользовательские параметры"):
        # 1. Perturb the full test set to obtain statistically rigorous metrics
        perturbed_test = []
        is_clean = (rotation == 0 and blur == 0 and brightness == 0 and noise == 0)
        
        if is_clean:
            X_eval = self.X_test.astype(np.float32) / 255.0
        else:
            for img in self.X_test:
                perturbed_test.append(perturb_image(img, rotation, blur, brightness, noise))
            X_eval = np.array(perturbed_test, dtype=np.float32) / 255.0

        # Run batched inference
        preds1 = self.model_clean.predict(X_eval, batch_size=128, verbose=0)
        preds2 = self.model_aug.predict(X_eval, batch_size=128, verbose=0)
        
        top_idx1 = np.argmax(preds1, axis=1)
        top_idx2 = np.argmax(preds2, axis=1)
        
        acc1_total = float(np.mean(top_idx1 == self.y_test) * 100)
        acc2_total = float(np.mean(top_idx2 == self.y_test) * 100)
        conf1_mean = float(np.mean(np.max(preds1, axis=1)) * 100)
        conf2_mean = float(np.mean(np.max(preds2, axis=1)) * 100)
        
        # 2. Evaluate per-category samples for visual cards
        categories_data = []
        rescued_count = 0
        
        for code in self.codes:
            cat_info = self.sample_per_category[code]
            cls_idx = cat_info['idx']
            base_img = cat_info['img_rgb']
            
            # Apply perturbation to the visual sample
            visual_img = perturb_image(base_img, rotation, blur, brightness, noise) if not is_clean else base_img
            sample_tensor = np.expand_dims(visual_img.astype(np.float32) / 255.0, axis=0)
            
            # Predict
            p1 = self.model_clean.predict(sample_tensor, verbose=0)[0]
            p2 = self.model_aug.predict(sample_tensor, verbose=0)[0]
            
            p1_top = int(np.argmax(p1))
            p1_conf = float(p1[p1_top]) * 100
            p1_correct = (p1_top == cls_idx)
            
            p2_top = int(np.argmax(p2))
            p2_conf = float(p2[p2_top]) * 100
            p2_correct = (p2_top == cls_idx)
            
            # Calculate class-level accuracy across all test samples for this class
            cls_mask = (self.y_test == cls_idx)
            cls_total = int(np.sum(cls_mask))
            cls_acc1 = float(np.mean(top_idx1[cls_mask] == cls_idx) * 100) if cls_total > 0 else 0.0
            cls_acc2 = float(np.mean(top_idx2[cls_mask] == cls_idx) * 100) if cls_total > 0 else 0.0
            
            # Status tag
            if not p1_correct and p2_correct:
                status = "rescued"  # Model 2 saved the case!
                rescued_count += 1
            elif p1_correct and p2_correct:
                status = "both_correct"
            elif not p1_correct and not p2_correct:
                status = "both_wrong"
            else:
                status = "m1_only"
                
            categories_data.append({
                "code": code,
                "ru_name": cat_info["ru_name"],
                "class_idx": cls_idx,
                "visual_img": visual_img,
                "m1_pred_idx": p1_top,
                "m1_pred_name": CLASS_NAMES_RU.get(self.codes[p1_top], self.idx_to_name[p1_top]),
                "m1_conf": p1_conf,
                "m1_correct": p1_correct,
                "m1_cls_acc": cls_acc1,
                "m2_pred_idx": p2_top,
                "m2_pred_name": CLASS_NAMES_RU.get(self.codes[p2_top], self.idx_to_name[p2_top]),
                "m2_conf": p2_conf,
                "m2_correct": p2_correct,
                "m2_cls_acc": cls_acc2,
                "status": status,
                "acc_gain": cls_acc2 - cls_acc1,
                "p1_top3": [(CLASS_NAMES_RU.get(self.codes[i], self.idx_to_name[i]), float(p1[i]) * 100) for i in np.argsort(p1)[-3:][::-1]],
                "p2_top3": [(CLASS_NAMES_RU.get(self.codes[i], self.idx_to_name[i]), float(p2[i]) * 100) for i in np.argsort(p2)[-3:][::-1]]
            })

        summary = {
            "preset_title": preset_title,
            "total_test_samples": len(self.X_test),
            "m1_accuracy": acc1_total,
            "m2_accuracy": acc2_total,
            "accuracy_delta": acc2_total - acc1_total,
            "m1_error_rate": 100.0 - acc1_total,
            "m2_error_rate": 100.0 - acc2_total,
            "m1_mean_conf": conf1_mean,
            "m2_mean_conf": conf2_mean,
            "rescued_count": rescued_count,
            "total_categories": len(self.codes),
            "categories": categories_data
        }
        return summary
