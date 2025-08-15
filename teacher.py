import importlib
from typing import Optional, Dict, Any, Callable
import torch
import torch.nn as nn
import torchvision.models as tvm

def resolve_factory(spec:str) -> Callable[[], nn.Module]:
    # spec can be "torchvision:resnet18" or "mypkg.module:factory"
    if spec.startswith("torchvision:"):
        name = spec.split(":")[1]
        if not hasattr(tvm, name):
            raise ValueError(f"torchvision has no model {name}")
        def factory():
            m = getattr(tvm, name)(weights=None, num_classes=10)
            return m
        return factory
    # python path
    module, func = spec.split(":")
    mod = importlib.import_module(module)
    factory = getattr(mod, func)
    return factory

def load_teacher(teacher:str=None, teacher_ckpt:str=None, device:str="cpu") -> nn.Module:
    if teacher_ckpt is not None and teacher is None:
        obj = torch.load(teacher_ckpt, map_location=device)
        if isinstance(obj, nn.Module):
            return obj.eval().to(device)
        elif isinstance(obj, dict) and "state_dict" in obj and "factory" in obj:
            m = resolve_factory(obj["factory"])()
            m.load_state_dict(obj["state_dict"])
            return m.eval().to(device)
        else:
            raise ValueError("Unknown checkpoint format")
    elif teacher is not None:
        m = resolve_factory(teacher)()
        return m.eval().to(device)
    else:
        raise ValueError("Provide --teacher or --teacher-ckpt")

@torch.no_grad()
def observe(teacher_model: nn.Module, dataloader, device="cpu", save_path:str="traces.pt", limit:int=None, return_features:bool=False):
    teacher_model.eval()
    all_logits, all_targets, all_feats = [], [], []
    n=0
    for x,y in dataloader:
        x = x.to(device)
        if return_features and hasattr(teacher_model, "forward"):
            out = teacher_model(x)
            if isinstance(out, tuple) and len(out)==2:
                logits, feats = out
            else:
                logits = out
                feats = None
        else:
            logits = teacher_model(x)
            feats = None
        all_logits.append(logits.cpu())
        all_targets.append(y)
        if feats is not None:
            if isinstance(feats, dict) and "feat" in feats:
                all_feats.append(feats["feat"].cpu())
        n += x.size(0)
        if limit is not None and n>=limit:
            break
    traces = {"logits": torch.cat(all_logits)[:limit], "targets": torch.cat(all_targets)[:limit]}
    if all_feats:
        traces["features"] = torch.cat(all_feats)[:limit]
    torch.save(traces, save_path)
    return save_path
