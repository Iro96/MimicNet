from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@dataclass
class TransGenome:
    # simple ViT-like
    patch: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    pool: str = "cls"  # or 'mean'

def make_default_genome(size:str, img_size:int, in_channels:int, n_classes:int) -> TransGenome:
    if size == "small":
        return TransGenome(patch=4, dim=128, depth=4, heads=4, mlp_dim=256, pool="cls")
    elif size == "medium":
        return TransGenome(patch=4, dim=256, depth=6, heads=8, mlp_dim=512, pool="cls")
    else:
        return TransGenome(patch=8, dim=384, depth=8, heads=8, mlp_dim=768, pool="cls")

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, dim, patch):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
    def forward(self, x):
        x = self.proj(x)  # B, D, H', W'
        x = x.flatten(2).transpose(1,2)  # B, N, D
        return x

class Transformer(nn.Module):
    def __init__(self, in_channels:int, n_classes:int, genome:TransGenome, img_size:int=32):
        super().__init__()
        self.genome = genome
        self.patch_embed = PatchEmbed(in_channels, genome.dim, genome.patch)
        num_patches = (img_size // genome.patch) * (img_size // genome.patch)
        self.cls_token = nn.Parameter(torch.zeros(1,1,genome.dim))
        self.pos = nn.Parameter(torch.randn(1, num_patches+1, genome.dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=genome.dim, nhead=genome.heads, dim_feedforward=genome.mlp_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=genome.depth)
        self.norm = nn.LayerNorm(genome.dim)
        self.fc = nn.Linear(genome.dim, n_classes)

    def forward(self, x, return_features=False):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos[:, :x.size(1)+1, :]
        x = self.encoder(x)
        x = self.norm(x)
        if self.genome.pool == "cls":
            feat = x[:,0]
        else:
            feat = x.mean(dim=1)
        logits = self.fc(feat)
        if return_features:
            return logits, {"feat": feat}
        return logits
