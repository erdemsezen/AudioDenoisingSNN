from types import SimpleNamespace

cfg = SimpleNamespace(
    sample_rate=16000,
    n_fft=512,
    hop_length=16,
    max_len=10000,  # This can be set to None to compute from dataset
    encode_mode="phased_rate",
    threshold=0.003,
    normalize=True,
    padding=True,
    n_epochs=2,
    batch_size=2,
    data_root="audio",
    n_freq_bins=257,
    n_iter=32
)
