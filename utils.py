import os
import json
import torch

def save_model(model, outdir, factory:str):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "student.pt")
    torch.save({"factory": factory, "state_dict": model.state_dict()}, path)
    return path
