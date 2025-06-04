import torch
import os
from datetime import datetime
from utils.encode import reconstruct_from_spikes
from utils.audio_utils import reconstruct_without_stretch
from models.builder import build_network
from data.dataloader import get_dataloaders
from utils.config import cfg
from utils.plot_utils import plot_stft_comparison
path = None
# Model yükle
snn = build_network(cfg)
latest_ckpt_folder = sorted(os.listdir("trained"))[-1]
model_path = path if path else os.path.join("Trained", latest_ckpt_folder)
print(f"Loading model from: {model_path}")
#snn.load_state_dict(torch.load(model_path))
snn.eval()

# Dataloader al
_, val_loader = get_dataloaders(cfg)
sample_batch = next(iter(val_loader))
input_spikes, target_spikes, clean_logstft, noisy_logstft, log_min, log_max, original_length, mask = sample_batch

# Girişleri hazırlama
input_spikes = input_spikes.permute(1, 0, 2)  # [T, B, F]

with torch.no_grad():
    snn(input_spikes)
    _, spike_out = list(snn.spk_rec.items())[-1]  # [T, B, F]
    spike_out = spike_out.permute(1, 0, 2)        # [B, T, F]

    T_real = int(mask[0].sum().item())
    trimmed_spike_out = spike_out[0][:T_real]
    trimmed_target_spikes = target_spikes[0][:T_real]

    pred_reconstructed = reconstruct_from_spikes(trimmed_spike_out, mode=cfg.encode_mode, trim=True)
    target_reconstructed = reconstruct_from_spikes(trimmed_target_spikes, mode=cfg.encode_mode, trim=True)

clean_logstft = clean_logstft[0][:T_real]
noisy_logstft = noisy_logstft[0][:T_real]

# Transpose to [F, T]
predicted_vis = pred_reconstructed.cpu().T
ground_truth_vis = target_reconstructed.cpu().T
clean_logstft_vis = clean_logstft.cpu().T
noisy_logstft_vis = noisy_logstft.cpu().T

pred_spikes = trimmed_spike_out.cpu().T
target_spikes_vis = trimmed_target_spikes.cpu().T

log_min_val = log_min[0].item()
log_max_val = log_max[0].item()

# === Klasör ismini oluştur ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
out_folder = f"outputs/wavs/{timestamp}_{cfg.encode_mode}_e{cfg.n_epochs}_len{cfg.max_len}_hop{cfg.hop_length}_nfft{cfg.n_fft}"
os.makedirs(out_folder, exist_ok=True)

# === WAV dosyalarını kaydet ===
reconstruct_without_stretch(clean_logstft_vis, log_min_val, log_max_val,
                            os.path.join(out_folder, "reconstructed_clean_STFT.wav"),
                            n_fft=cfg.n_fft, hop_length=cfg.hop_length, sample_rate=cfg.sample_rate, n_iter=cfg.n_iter)

reconstruct_without_stretch(noisy_logstft_vis, log_min_val, log_max_val,
                            os.path.join(out_folder, "reconstructed_noisy_STFT.wav"),
                            n_fft=cfg.n_fft, hop_length=cfg.hop_length, sample_rate=cfg.sample_rate, n_iter=cfg.n_iter)

reconstruct_without_stretch(predicted_vis, log_min_val, log_max_val,
                            os.path.join(out_folder, "reconstructed_predicted_STFT.wav"),
                            n_fft=cfg.n_fft, hop_length=cfg.hop_length, sample_rate=cfg.sample_rate, n_iter=cfg.n_iter)

reconstruct_without_stretch(ground_truth_vis, log_min_val, log_max_val,
                            os.path.join(out_folder, "reconstructed_target_STFT.wav"),
                            n_fft=cfg.n_fft, hop_length=cfg.hop_length, sample_rate=cfg.sample_rate, n_iter=cfg.n_iter)

plot_stft_comparison(
    out_folder,
    pred_spikes,
    target_spikes_vis,
    predicted_vis,
    ground_truth_vis,
    clean_logstft_vis,
    snn
)
print(f"WAV files saved to: {out_folder}")
print(f"Plots saved in: {out_folder}/plots")