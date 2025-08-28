import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.io as pio



# ======================
# β Schedule Functions
# ======================
def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1):
    steps = timesteps + 1
    x = torch.linspace(start, end, steps)
    betas = torch.sigmoid(x / tau)
    betas = (betas - betas.min()) / (betas.max() - betas.min()) * (0.02 - 0.0001) + 0.0001
    return betas[:-1]


# ======================
# Gaussian Diffusion
# ======================
class GaussianDiffusion(nn.Module):
    def __init__(self, model, image_size, channels=3, timesteps=1000, beta_schedule="linear", device="cuda"):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        self.device = device
        self.num_timesteps = timesteps

        # 选择 beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, timesteps)
        elif beta_schedule == "cosine":
            steps = timesteps + 1
            s = 0.008
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps + s) / (1 + s)) * torch.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = torch.clamp(betas, min=1e-5, max=0.999)
        elif beta_schedule == "sigmoid":
            betas = torch.linspace(-6, 6, timesteps)
            betas = torch.sigmoid(betas) * 0.02
        else:
            raise ValueError(f"Unknown beta_schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # 注册 buffer
        self.register_buffer("betas", betas.to(device))
        self.register_buffer("alphas", alphas.to(device))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(device))
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.to(device))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).to(device))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).to(device))
        self.register_buffer("sigma_t", torch.sqrt((1 - alphas_cumprod_prev) / (1 - alphas_cumprod) * (1 - alphas)).to(device))

    # 前向扩散：x0 -> xt
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    # 计算训练损失
    def p_losses(self, x_0, t):
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t)
        return F.mse_loss(predicted_noise, noise)

    # DDPM 推理采样
    @torch.inference_mode()
    def ddpm_sample(self, batch_size, sample_steps=None):
        if sample_steps is None:
            sample_steps = self.num_timesteps  # 默认全步
        step_indices = torch.linspace(0, self.num_timesteps-1, sample_steps, dtype=torch.long)

        x = torch.randn(batch_size, self.channels, self.image_size, self.image_size, device=self.device)
        for t in reversed(step_indices):
            t = t.item()
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            epsilon = self.model(x, t_batch)
            if t > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
            x = (1.0 / torch.sqrt(self.alphas[t])) * (
                x - ((1.0 - self.alphas[t]) / torch.sqrt(1.0 - self.alphas_cumprod[t])) * epsilon
            ) + self.sigma_t[t] * z
        return torch.clamp(x, 0.0, 1.0)
    
# ========== Base Trainer ==========
class Trainer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.train_losses = []
        self.valid_losses = []
        self.lrs = []

    def get_optimizer(self, lr: float):
        return Adam(self.model.parameters(), lr=lr)

    def train(
        self,
        train_loader,
        valid_loader=None,
        num_epochs: int = 10,
        device: torch.device = torch.device("cpu"),
        lr: float = 1e-3,
        warmup_epochs: int = 0,
        max_grad_norm: float = None,
        valid_every: int = 1,
    ):
        self.model.to(device)
        self.model.train()
        optimizer = self.get_optimizer(lr)
        warmup_steps = max(1, warmup_epochs * len(train_loader))
        scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)

        pbar = tqdm(range(num_epochs), desc="Epochs")
        for epoch in pbar:
            epoch_loss = 0.0
            for batch in train_loader:
                batch = tuple(b.to(device) for b in batch)
                optimizer.zero_grad()
                loss = self.train_step(batch)
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                self.lrs.append(optimizer.param_groups[0]['lr'])
                epoch_loss += loss.item() * len(batch[0])

            avg_train_loss = epoch_loss / len(train_loader.dataset)
            self.train_losses.append(avg_train_loss)

            # 验证
            if valid_loader is not None and epoch % valid_every == 0:
                self.model.eval()
                valid_loss = 0.0
                with torch.no_grad():
                    for batch in valid_loader:
                        batch = tuple(b.to(device) for b in batch)
                        loss = self.valid_step(batch)
                        valid_loss += loss.item() * len(batch[0])
                avg_valid_loss = valid_loss / len(valid_loader.dataset)
                self.valid_losses.append(avg_valid_loss)
                self.model.train()
            else:
                avg_valid_loss = None

            desc = f"Epoch {epoch+1}, train: {avg_train_loss:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}"
            if avg_valid_loss is not None:
                desc += f", valid: {avg_valid_loss:.4f}"
            pbar.set_postfix_str(desc)

        self.plot_metrics()
        # 训练结束后保存模型
        save_path = os.path.join(os.path.dirname(__file__), "trained_model.pth")
        torch.save(self.model.state_dict(), save_path)
        print(f"模型已保存到: {save_path}")

        self.plot_metrics()

    def train_step(self, batch):
        raise NotImplementedError

    def valid_step(self, batch):
        raise NotImplementedError

    def plot_metrics(self):
        pio.renderers.default = "browser"
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(y=self.train_losses, mode='lines', name='Train Loss'))
        fig1.update_layout(title="Train Loss", xaxis_title="Epoch", yaxis_title="Loss")
        fig1.show()

        if self.valid_losses:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=self.valid_losses, mode='lines+markers', name='Valid Loss', marker=dict(color='red')))
            fig2.update_layout(title="Valid Loss", xaxis_title="Epoch", yaxis_title="Loss")
            fig2.show()

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(y=self.lrs, mode='lines', name='Learning Rate'))
        fig3.update_layout(title="Learning Rate", xaxis_title="Step", yaxis_title="LR")
        fig3.show()



class DiffusionTrainer(Trainer):
    def __init__(self, diffusion, *args, **kwargs):
        super().__init__(diffusion.model)
        self.diffusion = diffusion

    def train_step(self, batch):
        x_0, _ = batch  # batch[0]: 图像
        t = torch.randint(0, self.diffusion.num_timesteps, (x_0.size(0),), device=x_0.device).long()
        return self.diffusion.p_losses(x_0, t)

    def valid_step(self, batch):
        x_0, _ = batch
        t = torch.randint(0, self.diffusion.num_timesteps, (x_0.size(0),), device=x_0.device).long()
        return self.diffusion.p_losses(x_0, t)
