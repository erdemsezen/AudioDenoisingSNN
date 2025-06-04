import torch
import torch.nn.functional as F


# -------------------------- SPIKE ENCODERS --------------------------

def delta_encode(spec: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Fires spikes when the signal changes beyond a threshold.
    """
    deltas = torch.zeros_like(spec)
    prev = torch.zeros(spec.shape[1])
    for t in range(spec.shape[0]):
        diff = spec[t] - prev
        deltas[t] = (diff.abs() > threshold).float() * diff
        prev = spec[t]
    return deltas


def deterministic_rate_encode(spec: torch.Tensor, time_window: int) -> torch.Tensor:
    """
    Fires spikes periodically based on signal intensity (inverse rate encoding).
    """
    T, F = spec.shape
    spikes = torch.zeros_like(spec)
    for t in range(T):
        for f in range(F):
            rate = spec[t, f].clamp(0, 1).item()
            period = int(1.0 / (rate + 1e-8))
            if t % max(1, period) == 0:
                spikes[t, f] = 1.0
    return spikes


def phased_deterministic_encode(spec: torch.Tensor, time_window: int) -> torch.Tensor:
    """
    Adds a phase offset to deterministic rate encoding to reduce synchrony.
    """
    F = spec.shape[1]
    phases = torch.rand(F)
    spikes = torch.zeros_like(spec)
    for t in range(time_window):
        theta = (t + phases) * spec.mean(0)  # broadcasting [F]
        spikes[t] = (theta % 1.0 < spec[t])
    return spikes


def sod_encode(spec: torch.Tensor, threshold: float, skip_initial: bool = True) -> torch.Tensor:
    """
    Send-on-Delta: emits ±1 when value changes beyond threshold.
    """
    T, F = spec.shape
    spikes = torch.zeros_like(spec)
    v_ref = spec[0].clone()

    for t in range(T):
        diff = spec[t] - v_ref

        up_idx = diff > threshold
        spikes[t, up_idx] = +1.0
        v_ref[up_idx] += threshold

        down_idx = diff < -threshold
        spikes[t, down_idx] = -1.0
        v_ref[down_idx] -= threshold

    if skip_initial:
        spikes[0].zero_()

    return spikes


def basic_encode(spec: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Fires spike if value exceeds threshold.
    """
    return (spec > threshold).float()


# ------------------------ SPIKE DECODING ------------------------

def reconstruct_from_spikes(spikes: torch.Tensor, mode: str = "delta",
                            smoothing_kernel_size: int = 31, threshold: float = 0.02,
                            mask: torch.Tensor = None, trim: bool = False) -> torch.Tensor:
    """
    Reconstruct approximation of the original signal from spike tensors.
    
    Args:
        spikes: [T, F] spike tensor
        mode: spike encoding mode ("delta", "sod", etc.)
        smoothing_kernel_size: for rate-based smoothing
        threshold: for SOD/basic scaling
        mask: optional [T] or [T, 1] mask indicating valid frames (1 = valid)
        trim: if True, returns only valid (unpadded) frames based on mask

    Returns:
        recon: [T, F] or [T_real, F] if trim=True
    """

    if mode == "delta":
        recon = spikes.cumsum(dim=0)

    elif mode == "sod":
        recon = spikes.cumsum(dim=0) * threshold

    elif mode == "basic":
        recon = spikes * threshold

    elif mode in ["rate", "phased_rate"]:
        time_steps, n_freq = spikes.shape
        kernel = torch.ones(n_freq, 1, smoothing_kernel_size, device=spikes.device) / smoothing_kernel_size
        spikes_reshaped = spikes.T.unsqueeze(0)  # [1, n_freq, T]
        smoothed = F.conv1d(spikes_reshaped, kernel, padding=smoothing_kernel_size // 2, groups=n_freq)
        recon = smoothed.squeeze(0).T  # [T, n_freq]

    else:
        raise ValueError(f"Unsupported reconstruction mode: {mode}")

    # Apply mask AFTER reconstruction
    if mask is not None:
        if mask.dim() == 1:
            mask = mask.unsqueeze(-1)  # [T] → [T, 1]
        elif mask.dim() == 2 and mask.shape[1] == 1:
            pass  # [T, 1] ok
        else:
            raise ValueError("Mask must be shape [T] or [T,1]")
        
        recon = recon * mask  # zero out padded frames

        if trim:
            T_real = int(mask.sum().item())
            recon = recon[:T_real]  # trim to valid length

    return recon



# -------------------------- MAIN ENCODER WRAPPER --------------------------

def spike_encode(
    stft_tensor: torch.Tensor,
    max_len: int = 1500,
    threshold: float = 0.002,
    normalize: bool = True,
    mode: str = "delta",
    padding: bool = True,
    global_min=None,
    global_max=None
) -> tuple[torch.Tensor, torch.Tensor, float, float, torch.Tensor]:
    """
    Encode with pre-padding + masking.
    Returns: spikes, padded_logstft, log_min, log_max, mask
    """

    # 1. Log-scale
    log_stft = torch.log(stft_tensor + 1e-6)  # [n_freq, T]

    # 2. Normalize
    if normalize:
        log_min = log_stft.min().item()
        log_max = log_stft.max().item()
        log_stft = (log_stft - log_min) / (log_max - log_min + 1e-8)
    else:
        log_min = None
        log_max = None

    log_stft = log_stft.T  # [T, n_freq]
    T_orig = log_stft.shape[0]

    # 3. Pad or truncate before encoding
    if padding:
        if T_orig < max_len:
            pad_len = max_len - T_orig
            log_stft = F.pad(log_stft, (0, 0, 0, pad_len), value=0.0)
        else:
            log_stft = log_stft[:max_len]
    T_padded = log_stft.shape[0]

    # 4. Generate mask for valid (non-padded) regions
    mask = torch.zeros(T_padded, dtype=torch.float32, device=log_stft.device)
    mask[:min(T_orig, max_len)] = 1.0  # 1 where original content exists

    # 5. Spike encode
    if mode == "delta":
        spikes = delta_encode(log_stft, threshold)
    elif mode == "rate":
        spikes = deterministic_rate_encode(log_stft, time_window=max_len)
    elif mode == "phased_rate":
        spikes = phased_deterministic_encode(log_stft, time_window=max_len)
    elif mode == "sod":
        spikes = sod_encode(log_stft, threshold)
    elif mode == "basic":
        spikes = basic_encode(log_stft, threshold)
    else:
        raise ValueError(f"Unsupported encoding mode: {mode}")

    return spikes, log_stft, log_min, log_max, mask

