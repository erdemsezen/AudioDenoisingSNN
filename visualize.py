import json
import os
from utils.plot_utils import plot_firing_rates, plot_loss

# (1) Kullanıcı path belirtmezse default olarak Trained/logs.json
path = "Trained/2025-05-30_17-09_phased_rate_e1_len3/logs.json"

log_path = path if path else "Trained/logs.json"

if not os.path.exists(log_path):
    raise FileNotFoundError(f"`{log_path}` bulunamadı.")

print(f"Loading logs from: {log_path}")

# (2) logs.json yükle
with open(log_path, "r") as f:
    logs = json.load(f)

# (3) logs.json’un bulunduğu klasörü al
log_dir = os.path.dirname(log_path)
plots_dir = os.path.join(log_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)

# (4) Grafik çizimlerini bu klasöre kaydet
plot_firing_rates(logs, save_path=os.path.join(plots_dir, "firing_rates.png"))
plot_loss(logs, save_path=os.path.join(plots_dir, "loss.png"))

print(f"Plots saved in: {plots_dir}")
