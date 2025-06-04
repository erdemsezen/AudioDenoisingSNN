from utils.encode import spike_encode
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import torch
import os

class SpikeSpeechEnhancementDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, sample_rate=16000, n_fft=1024, hop_length=256,
                 max_len=1500, threshold=0.02, mode="phased_rate",normalize=True,padding=True):
        
        self.noisy_paths = sorted([os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir) if f.endswith('.wav')])
        self.clean_paths = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.wav')])
        
        self.stft_transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=1.0
        )
        self.mode = mode
        self.threshold = threshold
        self.normalize = normalize
        self.padding = padding

        if max_len is None:
            print("Computing max_len from dataset...")
            self.max_len = self._compute_max_len()
            print(f"Computed max_len = {self.max_len}")
        else:
            self.max_len = max_len

    def __getitem__(self, idx):
        noisy_wav, _ = torchaudio.load(self.noisy_paths[idx])
        clean_wav, _ = torchaudio.load(self.clean_paths[idx])
        original_length = clean_wav.shape[-1]

        # 1) Compute STFT magnitude spectrograms [F, T]
        noisy_spec = self.stft_transform(noisy_wav).squeeze(0)
        clean_spec = self.stft_transform(clean_wav).squeeze(0)
        normalize=self.normalize
        padding=self.padding

        # 2) Spike encode (log + normalize inside the encoder)
        noisy_spikes, noisy_normed,log_min,log_max,mask_noisy = spike_encode(
            stft_tensor=noisy_spec,
            max_len=self.max_len,
            threshold=self.threshold,
            normalize=normalize,
            mode=self.mode,
            padding=padding
        )

        clean_spikes, clean_normed,log_min,log_max,mask_clean = spike_encode(
            stft_tensor=clean_spec,
            max_len=self.max_len,
            threshold=self.threshold,
            normalize=normalize,
            mode=self.mode,
            padding=padding
        )

        if normalize==False:
            log_min = None
            log_max = None
        
        if mask_clean.shape[0] != mask_noisy.shape[0]:
            print("Mask lengths are not equal")

        return noisy_spikes, clean_spikes, clean_normed, noisy_normed,log_min,log_max,original_length,mask_clean

    def __len__(self):
        return len(self.noisy_paths)
    
    def _compute_max_len(self):
        max_len = 0
        for path in self.clean_paths:
            wav, _ = torchaudio.load(path)
            spec = self.stft_transform(wav).squeeze(0)  # [F, T]
            T_len = spec.shape[-1]  # Number of time frames
            if T_len > max_len:
                max_len = T_len
        return max_len