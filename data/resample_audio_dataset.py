import os
import torchaudio
import torchaudio.transforms as T

# Define paths
data_root = r"E:/VSProjects/datasets/audioVCTK/" #C:/VSProjects/spiking-fpga-project/audio
orig_dirs = ["noisy_trainset_28spk_wav", "clean_trainset_28spk_wav"]
target_sr = 16000
orig_sr = 48000

# Define resampler
resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)

for split in orig_dirs:
    input_dir = os.path.join(data_root, split)
    output_dir = os.path.join(data_root, f"{split}_{target_sr}")

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".wav"):
            continue

        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)

        waveform, sr = torchaudio.load(in_path)
        print(f"Loaded {filename} with sample rate {sr}")

        if sr != target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        torchaudio.save(out_path, waveform, sample_rate=target_sr)


        print(f"Resampled {filename} → {target_sr} Hz → saved to {out_path}")
