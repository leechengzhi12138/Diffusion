import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
class DataModule(nn.Module):
    """
    通用 DataModule，定义统一接口
    """
    def __init__(self, batch_size=64, val_split=0.2, root="./data", transform=None, dataset_cls=None):
        super().__init__()
        self.batch_size = batch_size
        self.val_split = val_split
        self.root = root
        self.transform = transform
        self.dataset_cls = dataset_cls

        # 用于决定 device（CPU/GPU）
        self.register_buffer("dummy", torch.zeros(1))

        # 数据集占位符
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None

    def setup(self):
        if self.dataset_cls is None:
            raise ValueError("dataset_cls must be provided by subclass.")

        # 下载并加载训练数据
        full_train = self.dataset_cls(root=self.root, train=True, download=True, transform=self.transform)
        val_size = int(len(full_train) * self.val_split)
        train_size = len(full_train) - val_size

        self.train_dataset, self.valid_dataset = random_split(full_train, [train_size, val_size])

        # 下载并加载测试数据
        self.test_dataset = self.dataset_cls(root=self.root, train=False, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def valid_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def sample(self, num_samples: int, split="train"):
        if split == "train":
            dataset = self.train_dataset
        elif split == "valid":
            dataset = self.valid_dataset
        elif split == "test":
            dataset = self.test_dataset
        else:
            raise ValueError(f"Unknown split: {split}")

        # 随机取样
        indices = torch.randperm(len(dataset))[:num_samples]
        samples, labels = zip(*[dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy.device)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels
    
class MNISTDataModule(DataModule):
    def __init__(self, batch_size=256, val_split=0.2, root="./data"):
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        super().__init__(batch_size=batch_size, val_split=val_split, root=root, transform=transform, dataset_cls=datasets.MNIST)