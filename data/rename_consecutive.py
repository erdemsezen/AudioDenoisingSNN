import os

# Path to the folder
folder_path = "C:/VSProjects/spiking-fpga-project/audio/noisy"

# List and sort all .wav files
files = sorted([f for f in os.listdir(folder_path) if f.endswith(".wav")])

# Rename files to 000.wav, 001.wav, ...
for i, filename in enumerate(files):
    new_name = f"{i:03d}.wav"
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, new_name)
    os.rename(src, dst)
    print(f"Renamed: {filename} -> {new_name}")
