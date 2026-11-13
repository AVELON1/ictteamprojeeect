import os
import json
import matplotlib.pyplot as plt
import numpy as np

os.makedirs('assets', exist_ok=True)

# 1. Benchmark Bar Chart
categories = ['Clean Test Set', 'Adverse / Distorted Set']
model_clean_acc = [99.26, 72.43]
model_aug_acc = [99.88, 88.50]

x = np.arange(len(categories))
width = 0.32

fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#1e293b')

rects1 = ax.bar(x - width/2, model_clean_acc, width, label='Model 1 (Clean Baseline)', color='#f43f5e', edgecolor='none')
rects2 = ax.bar(x + width/2, model_aug_acc, width, label='Model 2 (Option B: OpenCV Enriched)', color='#10b981', edgecolor='none')

ax.set_ylabel('Accuracy (%)', color='#f8fafc', fontsize=12, fontweight='bold')
ax.set_title('Robustness Benchmark: Clean vs. Adversely Distorted Data', color='#f8fafc', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(categories, color='#f8fafc', fontsize=11, fontweight='600')
ax.set_ylim(50, 105)
ax.tick_params(colors='#94a3b8')
ax.grid(axis='y', linestyle='--', alpha=0.25, color='#cbd5e1')
ax.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc', fontsize=10)

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                textcoords="offset points", ha='center', va='bottom', color='#f43f5e', fontweight='bold', fontsize=10)

for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                textcoords="offset points", ha='center', va='bottom', color='#10b981', fontweight='bold', fontsize=10)

plt.tight_layout()
chart_path = 'assets/benchmark_comparison.png'
plt.savefig(chart_path, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print(f"Saved benchmark chart to {chart_path}")
