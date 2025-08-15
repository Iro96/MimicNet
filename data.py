from typing import Tuple
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

def make_dataloaders(dataset:str="mnist", batch_size:int=128, num_workers:int=2, limit:int=None) -> Tuple[DataLoader, DataLoader, int, int]:
    dataset = dataset.lower()
    if dataset == "mnist":
        mean, std = (0.1307,), (0.3081,)
        transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        in_ch, n_classes = 1, 10
    elif dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        transform_train = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
        transform_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
        test  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
        in_ch, n_classes = 3, 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if limit is not None:
        train = torch.utils.data.Subset(train, list(range(min(limit, len(train)))))
        test  = torch.utils.data.Subset(test, list(range(min(limit, len(test)))))

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, in_ch, n_classes
