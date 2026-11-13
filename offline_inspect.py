import os
import json
import cv2
import numpy as np
import tensorflow as tf

def run_offline_inspection():
    with open('models/classes_map.json', 'r') as f:
        cmap = json.load(f)
    idx_to_name = cmap['idx_to_name']
    code_to_idx = cmap['code_to_idx']

    m1 = tf.keras.models.load_model('models/model_clean.keras')
    m2 = tf.keras.models.load_model('models/model_aug.keras')

    with open('test_unseen/manifest.json', 'r') as f:
        manifest = json.load(f)

    print("=" * 105)
    print("                      OFFLINE EDGE CASE BENCHMARK (DATA-CENTRIC AI: OPTION B)")
    print("=" * 105)
    header = f"{'Test Scenario':<32} {'True Class':<18} {'Model 1 (Clean)':<25} {'Model 2 (Option B)':<25} {'Status':<10}"
    print(header)
    print("-" * 105)

    for item in manifest:
        p = os.path.join('test_unseen', item['file'])
        img = cv2.imread(p)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_64 = cv2.resize(img_rgb, (64, 64)) / 255.0
        tensor = np.expand_dims(img_64, 0)

        p1 = m1.predict(tensor, verbose=0)[0]
        p2 = m2.predict(tensor, verbose=0)[0]

        top1 = int(np.argmax(p1))
        top2 = int(np.argmax(p2))

        true_idx = code_to_idx[item['code']]
        true_name = idx_to_name[str(true_idx)]

        res1_str = f"{idx_to_name[str(top1)][:16]} ({p1[top1]*100:.1f}%)"
        res2_str = f"{idx_to_name[str(top2)][:16]} ({p2[top2]*100:.1f}%)"

        if top1 != true_idx and top2 == true_idx:
            status = "M1 FAILED"
        elif top1 == true_idx and top2 == true_idx:
            status = "BOTH OK"
        elif top1 != true_idx and top2 != true_idx:
            status = "EXTREME"
        else:
            status = "AUG ROBUST"

        scen_short = item['title'].split(' - ')[-1][:30]
        print(f"{scen_short:<32} {true_name[:16]:<18} {res1_str:<25} {res2_str:<25} {status:<10}")

    print("=" * 105)

if __name__ == '__main__':
    run_offline_inspection()
