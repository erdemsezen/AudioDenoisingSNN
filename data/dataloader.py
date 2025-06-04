import torch
from torch.utils.data import DataLoader
from data.dataset import SpikeSpeechEnhancementDataset

def get_dataloaders(cfg):
    dataset = SpikeSpeechEnhancementDataset(
        noisy_dir=f"{cfg.data_root}/noisy",
        clean_dir=f"{cfg.data_root}/clean",
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        max_len=cfg.max_len,
        threshold=cfg.threshold,
        normalize=cfg.normalize,
        mode=cfg.encode_mode,
        padding=cfg.padding,
    )
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    return DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True), DataLoader(val_set, batch_size=cfg.batch_size)

def get_validation_sample(cfg):
    _, val_loader = get_dataloaders(cfg)
    sample_batch = next(iter(val_loader))
    input_spikes = sample_batch[0].permute(1, 0, 2)
    return {
        "input_spikes": input_spikes,
        "log_min": sample_batch[4][0].item(),
        "log_max": sample_batch[5][0].item()
    }
