import os
import torch
import torchaudio
import torchaudio.transforms as T

# === CONFIGURATION ===
clean_dir = r"C:/VSProjects/spiking-fpga-project/audio/clean_22050"
sample_rate = 22050
n_fft = 1024
hop_length = 256
n_mels = 80
f_min = 0
f_max = 8000

# === MEL EXTRACTOR (must match HiFi-GAN) ===
mel_transform = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=n_fft,
    hop_length=hop_length,
    f_min=f_min,
    f_max=f_max,
    n_mels=n_mels,
    power=1.0,
    normalized=False
)

# === SCAN FILES ===
global_min = float('inf')
global_max = float('-inf')

for filename in os.listdir(clean_dir):
    if not filename.lower().endswith(".wav"):
        continue

    path = os.path.join(clean_dir, filename)
    wav, sr = torchaudio.load(path)

    if sr != sample_rate:
        resample = T.Resample(orig_freq=sr, new_freq=sample_rate)
        wav = resample(wav)

    mel = mel_transform(wav).squeeze(0)         # [n_mels, T]
    logmel = torch.log(mel + 1e-6).T             # [T, n_mels]

    global_min = min(global_min, logmel.min().item())
    global_max = max(global_max, logmel.max().item())

    print(f"Processed {filename} | min={logmel.min().item():.2f}, max={logmel.max().item():.2f}")

# === SAVE ===
torch.save({"min": global_min, "max": global_max}, "logmel_stats.pt")
print("\n✅ Saved global log-mel min/max to logmel_stats.pt")
print(f"Global logmel_min = {global_min:.4f}")
print(f"Global logmel_max = {global_max:.4f}")
