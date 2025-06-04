import numpy as np
import matplotlib.pyplot as plt

# Metric names
metrics = ['PESQ', 'STOI', 'SISDR', 'IntelSISNR', 'P808', 'OVRL', 'SIG', 'BAK']

# Veriler (sıralı)
dfn_scores = [2.336129, 0.91134708, 13.97896, 13.99431, 3.743795, 3.170853, 3.439706, 4.088344]
fsb_scores = [2.042506, 0.8931437, 14.41883, 14.40968, 3.640053, 3.047918, 3.35244, 3.970883]

# Plot setup
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, dfn_scores, width, label='DFN', color='#1f77b4')
bars2 = ax.bar(x + width/2, fsb_scores, width, label='FSB', color='#ff7f0e')

# Labels
ax.set_ylabel('Score')
ax.set_title('Metric Comparison: DFN vs FSB')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# Annotate bar values
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('metric_comparison.png')
