from typing import Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, T:float=4.0, alpha:float=0.7, targets=None):
    # standard KD (Hinton)
    if teacher_logits is None:
        ce = F.cross_entropy(student_logits, targets)
        return ce
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    kd = F.kl_div(p_s, p_t, reduction="batchmean") * (T*T)
    ce = F.cross_entropy(student_logits, targets) if targets is not None else 0.0
    return alpha * kd + (1 - alpha) * ce

@torch.no_grad()
def accuracy(logits, targets):
    pred = logits.argmax(dim=1)
    return (pred==targets).float().mean().item()
