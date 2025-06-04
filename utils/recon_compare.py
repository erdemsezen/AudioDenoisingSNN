import os
import re
import torch
import torchaudio
import pandas as pd
from pesq import pesq
from pystoi import stoi
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def compute_metrics(clean, recon, sr=16000):
    clean = clean[:recon.shape[-1]]
    recon = recon[:clean.shape[-1]]

    # Flatten to 1D (in case it's [1, T] instead of [T])
    clean = clean.squeeze()
    recon = recon.squeeze()

    pesq_score = pesq(sr, clean.numpy(), recon.numpy(), 'wb')
    stoi_score = stoi(clean.numpy(), recon.numpy(), sr, extended=False)
    return pesq_score, stoi_score


# === Paths
clean_dir = "audio/clean_16000"
recon_root = "audio/compare_recon"
output_xlsx = "speech_quality_results.xlsx"

# === Folder pattern: match mode, hop, fft
folder_pattern = re.compile(r"(delta|phased_rate)_Hop=(\d+)_Length=\d+_NFFT=(\d+)")

results = []

# === Loop over reconstructed folders
for subdir in os.listdir(recon_root):
    match = folder_pattern.match(subdir)
    if not match:
        print(f"⚠️ Skipping unrecognized folder: {subdir}")
        continue

    mode, hop, fft = match.groups()
    hop = int(hop)
    fft = int(fft)
    recon_dir = os.path.join(recon_root, subdir)

    for file in os.listdir(recon_dir):
        # Match files like reconstructed_0.wav
        match_index = re.match(r"reconstructed_(\d+)\.wav", file)
        if not match_index:
            print(f"⚠️ Skipping unexpected filename: {file}")
            continue

        index = int(match_index.group(1))
        clean_file = f"{index:03d}.wav"  # Pad to match clean files like 001.wav
        clean_path = os.path.join(clean_dir, clean_file)
        recon_path = os.path.join(recon_dir, file)

        if not os.path.exists(clean_path):
            print(f"⚠️ Missing clean file: {clean_file}")
            continue

        try:
            clean_wav, sr = torchaudio.load(clean_path)
            recon_wav, _ = torchaudio.load(recon_path)
            pesq_score, stoi_score = compute_metrics(clean_wav, recon_wav, sr)
            results.append({
                "mode": mode,
                "hop_length": hop,
                "fft_size": fft,
                "file": clean_file,
                "PESQ": pesq_score,
                "STOI": stoi_score,
            })
            print(f"✅ Compared: {clean_file} ↔ {file}")
        except Exception as e:
            print(f"[ERROR] {file} in {subdir}: {e}")

# === Save results to Excel
df = pd.DataFrame(results)

if df.empty:
    print("❌ No results to save. Check file paths or audio content.")
else:
    df.sort_values(by=["mode", "hop_length", "fft_size", "file"], inplace=True)
    df.to_excel(output_xlsx, index=False)
    print(f"✅ Results written to {output_xlsx}")
