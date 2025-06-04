from utils.encode import spike_encode
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import torch
import os

class SpikeSpeechEnhancementDatasetForRecon(Dataset):
    def __init__(self, clean_dir, sample_rate=16000, n_fft=1024, hop_length=256,
                 max_len=1500, threshold=0.02, mode="phased_rate",normalize=True,padding=True):
        
        
        self.clean_paths = sorted([os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.wav')])
        
        self.stft_transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=1.0
        )
        self.mode = mode
        self.max_len = max_len
        self.threshold = threshold
        self.normalize = normalize
        self.padding = padding

        if max_len is None:
            print("⏳ Computing max_len from dataset...")
            self.max_len = self._compute_max_len()
            print(f"✅ Computed max_len = {self.max_len}")
        else:
            self.max_len = max_len


    def __getitem__(self, idx):
        
        clean_wav, _ = torchaudio.load(self.clean_paths[idx])
        original_length = clean_wav.shape[-1]
        padding = self.padding

        # 1) Compute STFT magnitude spectrograms [F, T]
        
        clean_spec = self.stft_transform(clean_wav).squeeze(0)
        normalize=self.normalize

        clean_spikes, clean_normed,log_min,log_max = spike_encode(
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
        

        return clean_spikes, clean_normed, log_min,log_max,original_length,self.max_len

    def __len__(self):
        return len(self.clean_paths)
    
    def _compute_max_len(self):
        max_len = 0
        for path in self.clean_paths:
            wav, _ = torchaudio.load(path)
            spec = self.stft_transform(wav).squeeze(0)  # [F, T]
            T_len = spec.shape[-1]  # Number of time frames
            if T_len > max_len:
                max_len = T_len
        return max_len

