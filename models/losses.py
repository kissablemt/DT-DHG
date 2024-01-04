import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config_loader import LossConfigLoader

def bce_sigmoid(pos_score, neg_score):
    device = pos_score.device
    scores = torch.cat([pos_score, neg_score]).to(device)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)

    return F.binary_cross_entropy(torch.sigmoid(scores.ravel()), labels.ravel())

def bce(pos_score, neg_score):
    device = pos_score.device
    scores = torch.cat([pos_score, neg_score]).to(device)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)

    return F.binary_cross_entropy(scores.ravel(), labels.ravel())

def bce_logits(pos_score, neg_score, pos_weight=None):
    device = pos_score.device
    scores = torch.cat([pos_score, neg_score]).to(device)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).to(device)
    if pos_weight is not None:
        pos_weight = torch.tensor([pos_weight]).to(device)
    return F.binary_cross_entropy_with_logits(scores.ravel(), labels.ravel(), pos_weight=pos_weight)


def margin_loss(pos_score, neg_score):
    # 间隔损失 https://docs.dgl.ai/guide/training-link.html?highlight=link%20prediction
    n_edges = pos_score.shape[0]
    return (1 - pos_score.unsqueeze(1) + neg_score.view(n_edges, -1)).clamp(min=0).mean()


def hinge_loss(pos_score, neg_score):
    pos_loss = torch.nn.functional.relu(0 - pos_score).mean()
    neg_loss = torch.nn.functional.relu(-1 + neg_score).mean()
    loss = pos_loss + neg_loss
    return loss


class Loss:
    def __init__(self, config_path: str):
        config_loader = LossConfigLoader(config_path)
        self.config = config_loader.config


    def __call__(self, pos_score, global_neg_score, local_neg_score=None):
        loss = 0
        for cfg in self.config:
            func = globals()[cfg["name"]]
            args = cfg.get("args", {})
            if cfg["type"] == "global":
                loss += cfg.get("weight", 1) * func(pos_score, global_neg_score, **args)
            elif cfg["type"] == "local":
                if local_neg_score is None:
                    raise ValueError("Local negative score is None.")
                loss += cfg.get("weight", 1) * func(pos_score, local_neg_score, **args)
            else:
                raise ValueError("Unknown loss type.")

        return loss
    
