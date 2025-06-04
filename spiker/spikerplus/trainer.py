# trainer.py
import os
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import SpikePositionLoss
from utils.loss_for_delta import SpikePositionLossDelta
from encode import reconstruct_from_spikes

logging.basicConfig(level=logging.DEBUG)

def spike_hook(grad):
    print("⪡ ⪢ surrogate dL/dSpikes mean =", grad.abs().mean().item())
    return grad

def _print_hook(grad):
    # grad.shape == [T, B, F]
    print(f"⪡⪢ ∂L/∂S  mean = {grad.mean().item():.6f},  max = {grad.max().item():.6f}")
    # return grad if you want to modify it; here we just peek
    return grad

class Trainer:
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        loss_fn = None,
        loss_params: dict = None
        
        
    ):
        self.net         = net
        self.loss_params = loss_params or {}
        self.optimizer   = optimizer or torch.optim.Adam(net.parameters(), lr=3e-4)
        self.scheduler   = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.5
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        

        self.loss_fn = SpikePositionLoss(
            tau=5.0,
            lambda_pos=1.0,
            lambda_vr=0.01,   
            r_target=None,
            device=self.device
        )

        self.batch_losses = []

        logging.info(f"Initialized Trainer on {self.device}")
        self.logs = {
            'target_rate':    [],
            'pred_rate':      [],
            'spike_loss':     [],
            'rate_penalty':   [],
            'weight_change':  [],
        }

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader:   torch.utils.data.DataLoader,
        n_epochs:    int = 20,
        store:       bool = False,
        output_dir:  str = "Trained"
    ):
        start_time = time.time()
        logging.info(f"Training for {n_epochs} epochs")

        for epoch in range(1, n_epochs+1):
            logging.info(f"Epoch {epoch}/{n_epochs} started")
            self.batch_losses.clear()
            train_loss = self.train_one_epoch(train_loader)
            val_loss   = self.evaluate(val_loader)
            self.log(epoch, train_loss, val_loss, start_time)
            self.scheduler.step()

        if store:
            self.store(output_dir)

    def train_one_epoch(self, dataloader):
        self.net.train()
        total_loss = 0.0


        lambda_reg = 100  

        for batch_idx, (noisy_spikes, target_spikes, _,_,_,_,_,joint_mask) in enumerate(dataloader):
            # shapes: noisy_spikes [B, T, F]
            noisy = noisy_spikes.permute(1,0,2).to(self.device)   # [T, B, F]
            target = target_spikes.to(self.device)                # [B, T, F]
            mask = joint_mask.to(self.device)  # [B, T]



            r_target = target_spikes.gt(0).float().mean().item()
            

            print("Target firing-rate:", r_target)
            self.logs['target_rate'].append(r_target)

            self.optimizer.zero_grad()
            self.net(noisy)

            # final spike output
            _, rec    = list(self.net.spk_rec.items())[-1]      # [T, B, F]
            spk = rec
            spk.register_hook(_print_hook)
            spike_out = rec.permute(1,0,2)                      # [B, T, F]
            
            spike_out.register_hook(spike_hook)

            # compute loss
            spike_loss = self.loss_fn(spike_out, target, mask=mask)


            # raw MSE (optional)
            mse = F.mse_loss(spike_out, target.float())
            print("Raw MSE:", mse.item())
            print("========================")




            #regularization
            p = spike_out.mean()   # [0,1] arası float; gradyanı kopmuyor
            self.logs['spike_loss'].append(spike_loss.item())
                    

            loss = spike_loss
            
            loss.backward()
            for name, a in self.net.named_parameters():
                if a.grad is not None:
                    print(name, a.grad.norm().item())
       
            weight_deltas = {}

            w_pre = {}
            for name, param in self.net.named_parameters():
                if param.requires_grad and param.data is not None:
                    w_pre[name] = param.data.clone()
         
            
            self.optimizer.step()
            
            # Compute and log weight deltas
            for name, param in self.net.named_parameters():
                if param.requires_grad and param.data is not None:
                    delta = (param.data - w_pre[name]).norm().item()
                    weight_deltas[name] = delta
                    print(f"{name} change after step: {delta:.6f}")

            # Optionally store average or max delta in logs
            avg_delta = sum(weight_deltas.values()) / len(weight_deltas)
        
            if 'layer_deltas' not in self.logs:
                self.logs['layer_deltas'] = {}  # dict of lists

            for name, param in self.net.named_parameters():
                if param.requires_grad and param.data is not None:
                    delta = (param.data - w_pre[name]).norm().item()
                    if name not in self.logs['layer_deltas']:
                        self.logs['layer_deltas'][name] = []
                    self.logs['layer_deltas'][name].append(delta)


            # logging per batch
            spike_rate = (spike_out > 0).float().mean().item()
            logging.debug(
                f"[Batch {batch_idx}] SpikeRate: {p:.3f} | "
                f"SpkLoss: {spike_loss.item():.3f} | "
            )
            self.batch_losses.append(loss.item())
            total_loss += loss.item()
            self.logs['pred_rate'].append(p.item())

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def evaluate(self, dataloader):
        self.net.eval()
        total_loss = 0.0
        with torch.no_grad():
            for noisy_spikes, target_spikes, _,_,_,_,_,joint_mask in dataloader:
                noisy  = noisy_spikes.permute(1,0,2).to(self.device)
                target = target_spikes.to(self.device)

                self.net(noisy)
                _, rec    = list(self.net.spk_rec.items())[-1]
                spike_out = rec.permute(1,0,2)

                loss = self.loss_fn(spike_out, target)
                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def log(self, epoch, train_loss, val_loss, start_time=None):
        elapsed = time.time() - start_time if start_time else 0.0
        logging.info(
            f"Epoch {epoch} — Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Elapsed: {elapsed:.1f}s"
        )

    def store(self, out_dir, out_file="trained_state_dict.pt"):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, out_file)
        torch.save(self.net.state_dict(), path)
        logging.info(f"Model state dict saved to {path}")
