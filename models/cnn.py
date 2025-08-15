from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class CNNGenome:
    # list of conv blocks: (out_channels, kernel_size, stride)
    blocks: list
    classifier_hidden: int = 128

def make_default_genome(size:str, in_channels:int, n_classes:int) -> CNNGenome:
    size = size.lower()
    if size == "small":
        blocks = [(32,3,1),(64,3,1)]
        hidden = 128
    elif size == "medium":
        blocks = [(64,3,1),(128,3,1),(128,3,2)]
        hidden = 256
    else:
        blocks = [(64,3,1),(128,3,1),(256,3,2),(256,3,1)]
        hidden = 512
    return CNNGenome(blocks=blocks, classifier_hidden=hidden)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1):
        super().__init__()
        p = k//2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class CNNNet(nn.Module):
    def __init__(self, in_channels:int, n_classes:int, genome:CNNGenome):
        super().__init__()
        ch = in_channels
        layers = []
        for out_ch, k, s in genome.blocks:
            layers.append(ConvBlock(ch, out_ch, k, s))
            ch = out_ch
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(ch, genome.classifier_hidden)
        self.fc2 = nn.Linear(genome.classifier_hidden, n_classes)

    def forward(self, x, return_features=False):
        f = self.features(x)
        g = self.pool(f).flatten(1)
        h = F.relu(self.fc1(g))
        logits = self.fc2(h)
        if return_features:
            return logits, {"feat": g, "hidden": h}
        return logits
