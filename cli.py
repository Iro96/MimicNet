import os, json, time, random
import typer
from rich import print
from rich.table import Table
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .data import make_dataloaders
from .teacher import load_teacher, observe as observe_teacher
from .models.cnn import CNNGenome, CNNNet, make_default_genome as make_cnn_genome
from .models.transformer import TransGenome, Transformer, make_default_genome as make_trans_genome
from .distill import kd_loss, accuracy
from .evolution import init_population, mutate, crossover
from .utils import save_model

app = typer.Typer(add_completion=False)

def build_model(model_type:str, genome, in_ch:int, n_classes:int, img_size:int=32):
    if model_type=="cnn":
        return CNNNet(in_ch, n_classes, genome)
    else:
        return Transformer(in_ch, n_classes, genome, img_size=img_size)

def eval_model(model, dataloader, device, teacher_traces=None):
    model.eval()
    accs = []
    with torch.no_grad():
        for xb,yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            accs.append(accuracy(logits, yb))
    return sum(accs)/len(accs)

def train_one(model, train_loader, val_loader, device, epochs:int, lr:float, teacher_traces=None, T:float=4.0, alpha:float=0.7):
    model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_acc = 0.0
    for ep in range(1, epochs+1):
        model.train()
        total_loss=0.0
        for xb,yb in train_loader:
            xb,yb = xb.to(device), yb.to(device)
            logits = model(xb)
            if teacher_traces is not None:
                # use random slice of precomputed teacher logits
                idx = torch.randint(0, teacher_traces["logits"].shape[0], (yb.size(0),), device=yb.device)
                t_logits = teacher_traces["logits"].to(device)[idx]
            else:
                t_logits = None
            loss = kd_loss(logits, t_logits, T=T, alpha=alpha, targets=yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        va = eval_model(model, val_loader, device)
        print(f"[epoch {ep}] loss={total_loss/len(train_loader):.4f} val_acc={va:.4f}")
        if va>best_acc:
            best_acc=va
    return best_acc

@app.command()
def observe(
    dataset: str = typer.Option("cifar10", help="mnist or cifar10"),
    teacher: Optional[str] = typer.Option(None, help="e.g., torchvision:resnet18 or module:factory"),
    teacher_ckpt: Optional[str] = typer.Option(None, help="Path to a checkpoint containing a teacher."),
    batch_size: int = 128,
    limit: Optional[int] = typer.Option(None, help="Limit number of samples to trace."),
    save_traces: str = typer.Option("traces/teacher.pt", help="Where to save the trace file."),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
):
    train_loader, _, in_ch, n_classes = make_dataloaders(dataset, batch_size=batch_size, limit=limit)
    m = load_teacher(teacher, teacher_ckpt, device=device)
    os.makedirs(os.path.dirname(save_traces), exist_ok=True)
    path = observe_teacher(m, train_loader, device=device, save_path=save_traces, limit=limit, return_features=False)
    print(f"[bold green]Saved teacher traces to[/] {path}")

@app.command()
def train(
    dataset: str = "cifar10",
    type: str = typer.Option("cnn", help="cnn or transformer"),
    size: str = typer.Option("small", help="small/medium/large"),
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    teacher_traces: Optional[str] = typer.Option(None, help="Path to traces .pt file"),
    outdir: str = typer.Option("runs/train"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
):
    train_loader, val_loader, in_ch, n_classes = make_dataloaders(dataset, batch_size=batch_size)
    if type=="cnn":
        genome = make_cnn_genome(size, in_ch, n_classes)
    else:
        img_size = 28 if dataset=="mnist" else 32
        genome = make_trans_genome(size, img_size, in_ch, n_classes)
    model = build_model(type, genome, in_ch, n_classes, img_size=28 if dataset=="mnist" else 32)
    traces = torch.load(teacher_traces) if teacher_traces else None
    best = train_one(model, train_loader, val_loader, device, epochs, lr, traces)
    path = save_model(model, outdir, factory=f"nnwatcher.cli:autobuild_{type}")
    print(f"[bold green]Done.[/] Best val acc={best:.4f}. Saved to {path}")

def autobuild_cnn():
    # used by saved checkpoints to rebuild
    from .models.cnn import CNNNet, CNNGenome
    # fallback tiny
    return CNNNet(3,10,CNNGenome(blocks=[(64,3,1),(128,3,1)], classifier_hidden=128))

def autobuild_transformer():
    from .models.transformer import Transformer, TransGenome
    g = TransGenome(patch=4, dim=128, depth=4, heads=4, mlp_dim=256, pool="cls")
    return Transformer(3,10,g, img_size=32)

@app.command()
def evolve(
    dataset: str = "cifar10",
    type: str = typer.Option("cnn", help="cnn or transformer"),
    size: str = typer.Option("small"),
    epochs: int = 2,
    population: int = 8,
    generations: int = 5,
    batch_size: int = 128,
    lr: float = 1e-3,
    teacher_traces: Optional[str] = typer.Option(None, help="Path to traces .pt file"),
    outdir: str = typer.Option("runs/evolve"),
    device: str = typer.Option("cuda" if torch.cuda.is_available() else "cpu"),
    tournament_k: int = 3,
):
    train_loader, val_loader, in_ch, n_classes = make_dataloaders(dataset, batch_size=batch_size)
    img_size = 28 if dataset=="mnist" else 32
    pop = init_population(type, size, population, in_ch, n_classes, img_size=img_size)
    traces = torch.load(teacher_traces) if teacher_traces else None

    def score_ind(ind):
        model = build_model(type, ind.genome, in_ch, n_classes, img_size=img_size).to(device)
        acc = train_one(model, train_loader, val_loader, device, epochs, lr, traces)
        ind.score = acc
        return acc

    # initial scoring
    for i,ind in enumerate(pop):
        print(f"[bold blue]Scoring individual {i+1}/{len(pop)}[/]")
        score_ind(ind)

    for gen in range(1, generations+1):
        print(f"[magenta]=== Generation {gen}/{generations} ===[/]")
        # selection: tournament
        new_pop = []
        while len(new_pop)<len(pop):
            contestants = random.sample(pop, k=min(tournament_k, len(pop)))
            parent1 = max(contestants, key=lambda x:x.score)
            parent2 = max(random.sample(pop, k=min(tournament_k, len(pop))), key=lambda x:x.score)
            child_genome = crossover(parent1.genome, parent2.genome, type)
            # mutate
            if random.random()<0.9:
                child_genome = mutate(child_genome, type)
            new_pop.append(type(parent1).__class__(genome=child_genome))  # Individual

        # replace worst half with children
        all_inds = pop + new_pop
        # score children
        for i,ind in enumerate(new_pop):
            print(f"[bold yellow]Scoring child {i+1}/{len(new_pop)}[/]")
            score_ind(ind)
        all_inds.sort(key=lambda x:x.score, reverse=True)
        pop = all_inds[:population]
        print(f"Best this gen: {pop[0].score:.4f}")

    best = max(pop, key=lambda x:x.score)
    print(f"[bold green]Best overall val acc: {best.score:.4f}[/]")
    model = build_model(type, best.genome, in_ch, n_classes, img_size=img_size)
    path = save_model(model, outdir, factory=f"nnwatcher.cli:autobuild_{type}")
    print(f"Saved best model to {path}")

if __name__ == "__main__":
    app()
