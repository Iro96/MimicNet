<div align="center">
<h1>MimicNet</h1>
<p>
<strong>MimicNet</strong> is a research framework for building neural networks that <strong>learn by watching other neural networks</strong>.  
It enables a "student" model to observe a pretrained "teacher" model’s predictions and intermediate activations, learn via <strong>knowledge distillation</strong>, and <strong>evolve its own architecture</strong> over time using algorithms such as NEAT-inspired mutations.
</p>

[![Issues](https://img.shields.io/github/issues-raw/Iro96/MimicNet)](https://github.com/Iro96/MimicNet/issues)
[![Pytest](https://github.com/Iro96/MimicNet/actions/workflows/python-app.yml/badge.svg)](https://github.com/Iro96/Mimic/actions/workflows/python-app.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Iro96/MimicNet/blob/main/LICENSE)
![GitHub Repo stars](https://img.shields.io/github/stars/Iro96/MimicNet)

</div>



---

## Features
- **Multi-architecture support** — Create CNN or Transformer-based student models.
- **Teacher observation** — Capture logits and intermediate layer outputs from any PyTorch model.
- **Knowledge distillation** — Train students from teacher outputs using temperature scaling and adaptive loss weighting.
- **Evolutionary architecture search** — NEAT-like mutation and crossover to adapt student structures.
- **Dataset support** — MNIST and CIFAR-10 included; easily extendable to other datasets.
- **CLI interface** — Run `observe`, `train`, and `evolve` commands from the terminal.
- **Size-aware methods** — Choose `small`, `medium`, or `large` to avoid excessive memory or slow processing.

---

## *$* Installation


1. Clone the repository
```bash
git clone https://github.com/Iro96/MimicNet.git
cd MimicNet
```

2. Create and activate a virtual environment
```bash
python -m venv .venv
```

```bash
# On Windows
.\.venv\Scripts\Activate.ps1

# On Linux/MacOS
source .venv/bin/activate
```

3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
1. **Observe a teacher model**
```bash
python -m mimicnet.cli observe \
    --dataset cifar10 \
    --teacher torchvision:resnet18 \
    --save-traces traces/cifar_resnet18.pt \
    --limit 5000
```
> This collects CIFAR-10 samples, runs them through ResNet18, and stores the logits and intermediate activations.

2. **Train a student model**
```bash
python -m mimicnet.cli train \
    --dataset cifar10 \
    --type cnn \
    --size small \
    --epochs 3 \
    --teacher-traces traces/cifar_resnet18.pt \
    --outdir runs/cnn_small
```
> Trains a small CNN using knowledge distillation from the stored teacher traces.

3. **Evolve a student model**
```bash
python -m mimicnet.cli evolve \
    --dataset cifar10 \
    --type cnn \
    --size small \
    --epochs 2 \
    --population 6 \
    --generations 3 \
    --teacher-traces traces/cifar_resnet18.pt \
    --outdir runs/evolve_demo
```
> Performs a small NEAT-like evolutionary search to improve student performance.

---
### Contributing
> Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to modify.
