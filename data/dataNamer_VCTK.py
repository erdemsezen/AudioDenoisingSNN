import os
from pathlib import Path

clean_dir = Path(r"E:/VSProjects/datasets/audioVCTK/clean_trainset_28spk_wav")
noisy_dir = Path(r"E:/VSProjects/datasets/audioVCTK/noisy_trainset_28spk_wav")

# Dosyaları sırala
clean_files = sorted(clean_dir.glob("*.wav"))
noisy_files = sorted(noisy_dir.glob("*.wav"))

assert len(clean_files) == len(noisy_files), "Clean ve noisy klasörleri eşit sayıda dosya içermiyor!"

num_digits = len(str(len(clean_files)))  # kaç haneli olacak? 11571 => 5

for idx, (clean_path, noisy_path) in enumerate(zip(clean_files, noisy_files)):
    new_name = f"{str(idx).zfill(num_digits)}.wav"  # örn. "00000.wav"
    
    new_clean_path = clean_path.with_name(new_name)
    new_noisy_path = noisy_path.with_name(new_name)

    os.rename(clean_path, new_clean_path)
    os.rename(noisy_path, new_noisy_path)

print(f"Yeniden isimlendirme tamamlandı. Toplam dosya: {len(clean_files)}")
