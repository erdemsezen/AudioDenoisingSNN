import torch
import torch.nn.functional as F
from snntorch import spikegen

def delta_encode(mel: torch.Tensor, threshold: float) -> torch.Tensor:
    """Hand-rolled delta encoder (+1/–1 on jumps > threshold)."""
    deltas = torch.zeros_like(mel)
    prev = torch.zeros(mel.shape[1], device=mel.device)
    for t in range(mel.shape[0]):
        diff = mel[t] - prev
        deltas[t] = (diff.abs() > threshold).float() * diff
        prev = mel[t]
    return deltas


def deterministic_rate_encode(mel: torch.Tensor, time_window: int) -> torch.Tensor:
    """Hand-rolled deterministic rate encoder (evenly spaced)."""
    T, F_req = mel.shape
    spikes = torch.zeros_like(mel)
    for t in range(T):
        for f in range(F_req):
            rate = mel[t, f].clamp(0, 1).item()
            period = int(1.0 / (rate + 1e-8))
            if t % max(1, period) == 0:
                spikes[t, f] = 1.0
    return spikes


def sod_encode(mel: torch.Tensor, threshold: float, skip_initial: bool = True) -> torch.Tensor:
    """Send-On-Delta: ±1 when envelope moves ±threshold from last reference."""
    T, F_req = mel.shape
    spikes = torch.zeros_like(mel)
    v_ref = mel[0].clone()
    for t in range(T):
        diff = mel[t] - v_ref
        up = diff > threshold
        dn = diff < -threshold
        spikes[t, up] = +1.0; v_ref[up] += threshold
        spikes[t, dn] = -1.0; v_ref[dn] -= threshold
    if skip_initial:
        spikes[0].zero_()
    return spikes


def basic_encode(mel: torch.Tensor, threshold: float) -> torch.Tensor:
    """Hard threshold: a spike whenever mel > threshold."""
    return (mel > threshold).float()


def reconstruct_from_spikes(
    spikes: torch.Tensor,
    mode: str = "delta",
    smoothing_kernel_size: int = 11,
    threshold: float = 0.02
) -> torch.Tensor:
    """
    Decode spike trains back into analog/log-mel.
    """
    if mode in ("delta", "snn_delta"):
        return spikes.cumsum(dim=0)

    elif mode == "sod":
        return spikes.cumsum(dim=0) * threshold

    elif mode == "basic":
        return spikes * threshold

    elif mode in ("rate", "snn_rate"):
        # sliding-window average
        has_batch = False
        if spikes.ndim == 3:
            # [B,T,F] -> [B, F, T]
            spikes = spikes.permute(0, 2, 1)
            has_batch = True
        else:
            # [T, F] -> [1, F, T]
            spikes = spikes.T.unsqueeze(0)

        n_chans = spikes.shape[1]
        kernel = torch.ones(n_chans, 1, smoothing_kernel_size,
                            device=spikes.device) / smoothing_kernel_size
        out = F.conv1d(spikes, kernel,
                       padding=smoothing_kernel_size // 2,
                       groups=n_chans)

        if has_batch:
            return out.permute(0, 2, 1)
        return out.squeeze(0).T

    elif mode in ("latency", "snn_latency"):
        # first‐spike timing → analog
        # spikes: [T,F]
        T_req, F_req = spikes.shape
        first = torch.argmax(spikes.float(), dim=0)
        return 1.0 - first.float() / (T_req - 1)

    else:
        raise ValueError(f"Unknown reconstruction mode: {mode}")


def spike_encode(
    mel_tensor: torch.Tensor,
    max_len: int = 1500,
    threshold: float = 0.002,
    normalize: bool = True,
    mode: str = "delta"
):
    """
    Encode log-mel into spikes. Returns (spikes, mel_rescaled).
    """
    # 1) log + normalize + resize
    mel = torch.log(mel_tensor + 1e-6)
    if normalize:
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
    mel = F.interpolate(mel.unsqueeze(0),
                        size=max_len,
                        mode='linear',
                        align_corners=False
                        ).squeeze(0).T  # [T, F]

    # 2) pick encoder
    if mode == "delta":
        spikes = delta_encode(mel, threshold)
    elif mode == "rate":
        spikes = deterministic_rate_encode(mel, time_window=max_len)
    elif mode == "latency":
        from .encode import spike_encode_latency
        spikes, _ = spike_encode_latency(mel, max_len=max_len)
    elif mode == "sod":
        spikes = sod_encode(mel, threshold)
    elif mode == "basic":
        spikes = basic_encode(mel, threshold)

    # 3) snnTorch built-ins
    elif mode == "snn_rate":
        # per-channel rate encoding → [T, n_feats]
        T_req, n_feats = mel.shape
        spikes = torch.stack([
            spikegen.rate(mel[:, i], num_steps=T_req)
            for i in range(n_feats)
        ], dim=1)
    elif mode == "snn_latency":
        flat = mel.flatten()
        spikes_flat = spikegen.latency(flat, num_steps=max_len)
        spikes = spikes_flat.view(max_len, mel.shape[1])
    elif mode == "snn_delta":
        flat = mel.flatten()
        spikes_flat = spikegen.delta(flat, threshold=threshold)
        spikes = spikes_flat.view(max_len, mel.shape[1])

    else:
        raise ValueError(f"Unsupported encoding mode: {mode}")

    return spikes, mel


def blur_spikes(spk, k: int = 11):
    """
    Depth-wise 1-D average pool over time.
    Accepts [T, F]  or  [B, T, F]   →   returns same shape.
    """
    if spk.ndim == 2:                             # [T, F]
        spk_3d = spk.T.unsqueeze(0)               # → [1, F, T]
        squeeze_back = True
    elif spk.ndim == 3:                           # [B, T, F]
        spk_3d = spk.permute(0, 2, 1)             # → [B, F, T]
        squeeze_back = False
    else:
        raise ValueError("blur_spikes expects 2-D or 3-D tensor.")

    B, C, T = spk_3d.shape
    kernel   = torch.ones(C, 1, k, device=spk.device) / k          # [C,1,k]
    smoothed = F.conv1d(spk_3d, kernel, padding=k//2, groups=C)    # [B,C,T]

    if squeeze_back:                               # original was 2-D
        return smoothed.squeeze(0).T               # → [T, F]
    else:                                          # original was 3-D
        return smoothed.permute(0, 2, 1)           # → [B, T, F]
