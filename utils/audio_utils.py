import torch
import numpy as np
import soundfile as sf
from torchaudio.transforms import GriffinLim

def logstft_to_waveform(log_stft: torch.Tensor, n_fft: int = 1024, hop_length: int = 256,
                        win_length: int = None, sample_rate: int = 22050, n_iter: int = 32) -> torch.Tensor:
    if win_length is None:
        win_length = n_fft

    magnitude = torch.exp(log_stft) - 1e-6  # [F, T]
    griffinlim = GriffinLim(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=1.0,
        n_iter=n_iter
    )
    waveform = griffinlim(magnitude.unsqueeze(0))  # [1, T]
    return waveform


def reconstruct_without_stretch(
    logstft_tensor: torch.Tensor,
    log_min: float,
    log_max: float,
    filename: str,
    n_fft: int = 1024,
    hop_length: int = 256,
    sample_rate: int = 16000,
    n_iter: int = 32,
    original_length: int = None  # Accept original_length in samples
):
    # Denormalize
    denorm = logstft_tensor * (log_max - log_min) + log_min

    # Inverse STFT
    waveform = logstft_to_waveform(
        denorm, n_fft=n_fft, hop_length=hop_length,
        sample_rate=sample_rate, n_iter=n_iter
    ).squeeze(0).cpu().numpy()  # [T]

    # Trim to original number of samples (in waveform domain)
    if original_length is not None and original_length > 0:
        waveform = waveform[:original_length]

    # Save
    sf.write(filename, waveform, samplerate=sample_rate)
    return waveform

