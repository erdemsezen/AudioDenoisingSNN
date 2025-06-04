import os
import re
from shutil import copyfile

# === Klasör yolları ===
clean_dir = r"E:\VSProjects\datasets\audio\clean"
noisy_dir = r"E:\VSProjects\datasets\audio\noisy"

# === Dosyaların kopyalanacağı klasörler (isteğe bağlı) ===
output_clean = os.path.join(clean_dir, "renamed")
output_noisy = os.path.join(noisy_dir, "renamed")

os.makedirs(output_clean, exist_ok=True)
os.makedirs(output_noisy, exist_ok=True)

# === fileid çıkartma regex ===
fileid_pattern = re.compile(r"fileid_(\d+)\.wav")

def extract_fileids(file_list):
    ids = {}
    for fname in file_list:
        match = fileid_pattern.search(fname)
        if match:
            ids[match.group(1)] = fname
    return ids

# === Dosya listeleri ===
clean_files = [f for f in os.listdir(clean_dir) if f.endswith(".wav")]
noisy_files = [f for f in os.listdir(noisy_dir) if f.endswith(".wav")]

clean_ids = extract_fileids(clean_files)
noisy_ids = extract_fileids(noisy_files)

# === Ortak fileid'leri bul
common_ids = sorted(set(clean_ids.keys()) & set(noisy_ids.keys()), key=int)

print(f"[INFO] Eşleşen {len(common_ids)} dosya çifti bulundu.")

# === Yeniden adlandırma ve kopyalama
for idx, fid in enumerate(common_ids, start=1):
    clean_src = os.path.join(clean_dir, clean_ids[fid])
    noisy_src = os.path.join(noisy_dir, noisy_ids[fid])

    clean_dst = os.path.join(output_clean, f"clean_{idx}.wav")
    noisy_dst = os.path.join(output_noisy, f"noisy_{idx}.wav")

    copyfile(clean_src, clean_dst)
    copyfile(noisy_src, noisy_dst)

    print(f"[{idx}] {clean_ids[fid]} <-> {noisy_ids[fid]} → clean_{idx}.wav, noisy_{idx}.wav")

print("\n[OK] Tüm eşleşen dosyalar yeniden adlandırıldı.")
