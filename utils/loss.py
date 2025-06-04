import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikePositionLoss(nn.Module):
    def __init__(
        self,
        tau: float = 5.0,
        lambda_pos: float = 1.0,
        lambda_vr:  float = 0.01,
        r_target:    float = None,
        device: torch.device = None
    ):
        super().__init__()
        self.tau       = tau
        self.lambda_vr = lambda_vr
        self.lambda_pos = lambda_pos
        self.r_target  = r_target
        self.device    = device or torch.device("cpu")

        # Van Rossum kernel
        L = int(6 * tau)
        t_idx = torch.arange(0, L, device=self.device)
        kernel = torch.exp(-t_idx / tau).to(torch.float32)
        self.register_buffer("vr_kernel", kernel.view(1, 1, -1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        pred, target: [B, T, C]
        mask: [B, T]
        """
        B, T, C = pred.shape

        # ====== Position Loss ======
        if mask is not None:
            mask_exp = mask.unsqueeze(-1).expand(-1, -1, C)  # [B, T, C]
            pred_masked = pred[mask_exp == 1]                # [N_valid]
            target_masked = target[mask_exp == 1]            # [N_valid]
            pos_loss = F.binary_cross_entropy(pred_masked, target_masked.float())
        else:
            pos_loss = F.binary_cross_entropy(pred, target.float())

        # ====== Van Rossum Smoothing Loss ======
        p_in = pred.permute(0, 2, 1).reshape(B * C, 1, T)
        t_in = target.permute(0, 2, 1).reshape(B * C, 1, T)
        pad = self.vr_kernel.size(-1) // 2
        p_f = F.conv1d(p_in, self.vr_kernel, padding=pad)[..., :T]
        t_f = F.conv1d(t_in, self.vr_kernel, padding=pad)[..., :T]
        p_f = p_f.reshape(B, C, T).permute(0, 2, 1)
        t_f = t_f.reshape(B, C, T).permute(0, 2, 1)

        if mask is not None:
            vr_mask = mask.unsqueeze(-1).expand(-1, -1, C)  # [B, T, C]
            vr_loss = F.mse_loss(p_f[vr_mask == 1], t_f[vr_mask == 1])
        else:
            vr_loss = F.mse_loss(p_f, t_f)

        # ====== Total Loss ======
        loss = self.lambda_pos * pos_loss + self.lambda_vr * vr_loss
        return loss
