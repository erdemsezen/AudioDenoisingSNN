from models.builder import build_network
from spikerplus import Trainer
from data.dataloader import get_dataloaders
from utils.config import cfg

snn = build_network(cfg)
train_loader, val_loader = get_dataloaders(cfg)
trainer = Trainer(snn)
trainer.train(train_loader, val_loader, n_epochs=cfg.n_epochs, store=True, encode_mode=cfg.encode_mode, max_len=cfg.max_len)

