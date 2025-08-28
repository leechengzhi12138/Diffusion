import torch
from DataModule import MNISTDataModule
from Diffusion import GaussianDiffusion, DiffusionTrainer
from Model import UNet

# 初始化 DataModule
datamodule = MNISTDataModule(batch_size=128)
datamodule.setup()
train_loader = datamodule.train_dataloader()
valid_loader = datamodule.valid_dataloader()

# 初始化 UNet & Diffusion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


unet = UNet(
    source_channel=1,
    unet_base_channel=128,
    num_norm_groups=32,
).to(device)

diffusion = GaussianDiffusion(model=unet, image_size=32, channels=1, timesteps=1000, beta_schedule="linear", device=device)

# 使用新 Trainer
trainer = DiffusionTrainer(diffusion)
trainer.train(train_loader, valid_loader=valid_loader, num_epochs=50, device=device, lr=1e-4, valid_every=5)