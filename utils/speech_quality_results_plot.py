import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load Excel
df = pd.read_excel("speech_quality_results.xlsx")

# === Force numeric types and clean invalid rows
df["PESQ"] = pd.to_numeric(df["PESQ"], errors="coerce")
df["STOI"] = pd.to_numeric(df["STOI"], errors="coerce")
df = df.dropna(subset=["PESQ", "STOI"])

# === Group numeric metrics only
group_cols = ["mode", "hop_length", "fft_size"]
agg_cols = ["PESQ", "STOI"]
grouped = df[group_cols + agg_cols].groupby(group_cols)[agg_cols].mean().reset_index()

# === Color map: one color per mode
color_map = {
    "delta": "#1f77b4",         # blue
    "phased_rate": "#ff7f0e",   # orange
}

# === Plot loop
metrics = ["PESQ", "STOI"]
for metric in metrics:
    plt.figure(figsize=(12, 6))

    # Multi-line x labels
    x_labels = grouped.apply(
        lambda row: f"{row['mode']}\nhop={row['hop_length']}\nfft={row['fft_size']}",
        axis=1
    )

    scores = grouped[metric]
    modes = grouped["mode"]
    colors = [color_map[mode] for mode in modes]

    x = np.arange(len(x_labels))
    bars = plt.bar(x, scores, color=colors)

    plt.xticks(x, x_labels, rotation=0)
    plt.ylabel(metric)
    plt.title(f"Average {metric} by Configuration")
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()

    # Legend
    handles = [
        plt.Line2D([0], [0], color=color_map[m], lw=8) for m in color_map
    ]
    labels = [m for m in color_map]
    plt.legend(handles, labels, title="Encoding Mode")

    # Save and show
    plt.savefig(f"plot_{metric.lower()}_colored.png")
    print(f"✅ Saved: plot_{metric.lower()}_colored.png")
    plt.show()
