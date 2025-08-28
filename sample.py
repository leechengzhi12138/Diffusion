import torch
import os
from torchvision.utils import save_image
from Model import UNet
from Diffusion import GaussianDiffusion

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 加载模型
model = UNet(
    source_channel=1,
    unet_base_channel=128,
    num_norm_groups=32,
).to(device)
model_path = os.path.join(os.path.dirname(__file__), "trained_model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval

# 2. 初始化 GaussianDiffusion
diffusion = GaussianDiffusion(model=model, image_size=32, channels=1, timesteps=1000, beta_schedule="linear", device=device)

# 3. 定义不同采样步数
sampling_steps = [1000, 500, 250]  
batch_size = 4

output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)

for steps in sampling_steps:
    print(f"Sampling with {steps} steps...")
    samples = diffusion.ddpm_sample(batch_size=batch_size, sample_steps=steps)
    save_path = os.path.join(output_dir, f"generated_{steps}.png")
    save_image(samples, save_path, nrow=2)

    print(f"图片已保存到 {save_path}")
