from glob import glob
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from easydict import EasyDict as edict
import json
import models
from models import predictors as P
from models import losses as L
from utils import data_loader, dti_graph, sampler, config_loader, my_record, training_utils, base
from dgl.data.utils import save_graphs, load_graphs


def ogb_hits(y_pred_pos, y_pred_neg, k):
    '''
        from: https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py
    '''

    if len(y_pred_neg) < k:
        return {'hits@{}'.format(k): 1., "hits_list": torch.ones()}
    
    if isinstance(y_pred_pos, torch.Tensor):
        y_pred_pos = y_pred_pos.view(-1)
        y_pred_neg = y_pred_neg.view(-1)
        kth_score_in_negative_edges = torch.topk(y_pred_neg, k)[0][-1]
        hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

    # type_info is numpy
    else:
        y_pred_pos = y_pred_pos.reshape(-1)
        y_pred_neg = y_pred_neg.reshape(-1)
        kth_score_in_negative_edges = np.sort(y_pred_neg)[-k]
        hitsK = float(np.sum(y_pred_pos > kth_score_in_negative_edges)) / len(y_pred_pos)

    return hitsK


def ogb_mrr(y_pred_pos, y_pred_neg):
    '''
        from: https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py
    '''

    if isinstance(y_pred_pos, torch.Tensor):
        # calculate ranks
        y_pred_pos = y_pred_pos.view(-1, 1)
        # optimistic rank: "how many negatives have a larger score than the positive?"
        # ~> the positive is ranked first among those with equal score
        optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
        # pessimistic rank: "how many negatives have at least the positive score?"
        # ~> the positive is ranked last among those with equal score
        pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        mrr_list = 1./ranking_list.to(torch.float)
        
        # return edict({'mrr': mrr_list.mean(), 'mrr_list': mrr_list})
        return mrr_list.mean()

    else:
        y_pred_pos = y_pred_pos.reshape(-1, 1)
        optimistic_rank = (y_pred_neg >= y_pred_pos).sum(axis=1)
        pessimistic_rank = (y_pred_neg > y_pred_pos).sum(axis=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        mrr_list = 1./ranking_list.astype(np.float32)

        # return edict({'mrr': mrr_list.mean(), 'mrr_list': mrr_list})
        return mrr_list.mean()
    

def hit_at_k(pred, gt, k=10):
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt)
    # Sort the pred in descending order
    _, top_indices = torch.topk(pred, k=k, dim=1, largest=True, sorted=True)

    # Create a binary matrix with 1's at the top 10 recommended items
    top_pred = torch.zeros_like(pred)
    top_pred[torch.arange(top_indices.shape[0]).view(-1, 1), top_indices] = 1

    # Calculate the Hit@10 metric
    return torch.bitwise_and(top_pred.int(), gt.int()).sum() / gt.sum()

def pre_at_k(pred, gt, k=10):
    kth_score_in_all_edges = torch.topk(pred.flatten(), k)[0][-1]
    topK_adj = torch.where(pred >= kth_score_in_all_edges, 1, 0)
    preK = (torch.bitwise_and(gt, topK_adj).sum() / k).item()
    return preK

def compute(pos_score, neg_score):
    scores = torch.cat([pos_score.detach(), neg_score.detach()]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.cpu().shape[0]), torch.zeros(neg_score.cpu().shape[0])]
    ).numpy()
    
    precisions, recalls, thresholds = metrics.precision_recall_curve(labels, scores)
    aupr = metrics.auc(recalls, precisions)
    # aupr = compute_aupr(pos_score, neg_score)
    
    # 拿到最优结果以及索引
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])

    # 阈值
    best_threshold = thresholds[best_f1_score_index]

    # 阈值化
    preds = (scores >= best_threshold).astype(int) 

    f1 = metrics.f1_score(labels, preds)
    acc = metrics.accuracy_score(labels, preds)
    recall = metrics.recall_score(labels, preds)
    precision = metrics.precision_score(labels, preds)

    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    auc = metrics.auc(fpr, tpr)
    # auc = compute_auc(pos_score, neg_score)


    return {
        "aupr": aupr,
        "f1": f1,
        "accuracy": acc,
        "recall": recall,
        "precision": precision,
        "auc": auc,
        "best_threshold": best_threshold,
    }

def compute_aupr(pos_score, neg_score):
    scores = torch.cat([pos_score.detach(), neg_score.detach()]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.cpu().shape[0]), torch.zeros(neg_score.cpu().shape[0])]
    ).numpy()
    return metrics.average_precision_score(labels, scores)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score.detach(), neg_score.detach()]).cpu().numpy()
    labels = torch.cat(
        [torch.ones(pos_score.cpu().shape[0]), torch.zeros(neg_score.cpu().shape[0])]
    ).numpy()
    return metrics.roc_auc_score(labels, scores)

def load_result(k_th: int, res_dir: Union[Path, str], data_dir: Union[Path, str], skip_weight: bool = False):
    if isinstance(res_dir, str):
        res_dir = Path(res_dir)
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    args_json = res_dir / "args.json"
    if not args_json.exists():
        raise ValueError("args.json does not exist")
    with open(args_json, "r") as f:
        args = edict(json.load(f))
    
    base.setup_seed(seed=args.seed)

    # ============ Preparation ============
    data = data_loader.preparation(
        n_features=args.n_features, dataset_dir=data_dir / args.dataset, 
        top_thr=args.top_thr, not_rand_feat=args.not_rand_feat, device=args.device)

    # ============ Load Graph ============
    h = {
        "drug": torch.load(res_dir / "result" / f"{k_th}_drug_h.pt").to(args.device),
        "target": torch.load(res_dir / "result" / f"{k_th}_target_h.pt").to(args.device),
    }
    node_features = {
        "drug": torch.load(res_dir / "result" / f"drug_init.pt").to(args.device),
        "target": torch.load(res_dir / "result" / f"target_init.pt").to(args.device),
    }

    train_g = load_graphs(str(res_dir / "graph" / f"{k_th}_train_g.bin"))[0][0].to(args.device)
    test_g = load_graphs(str(res_dir / "graph" / f"{k_th}_test_g.bin"))[0][0].to(args.device)
    valid_g = load_graphs(str(res_dir / "graph" / f"{k_th}_valid_g.bin"))[0][0].to(args.device)
    train_neg_g = load_graphs(str(res_dir / "graph" / f"{k_th}_train_neg_g.bin"))[0][0].to(args.device)
    test_neg_g = load_graphs(str(res_dir / "graph" / f"{k_th}_test_neg_g.bin"))[0][0].to(args.device)
    valid_neg_g = load_graphs(str(res_dir / "graph" / f"{k_th}_valid_neg_g.bin"))[0][0].to(args.device)
    g = load_graphs(str(res_dir / "graph" / f"g.bin"))[0][0].to(args.device)
    g_dict = edict(train_g=train_g, test_g=test_g, 
                   valid_g=valid_g, valid_neg_g=valid_neg_g,
                   train_neg_g=train_neg_g, test_neg_g=test_neg_g, g=g)
    
    ret = edict(data=data, graphs=g_dict, 
                h=h, node_features=node_features, args=args)

    # ============ Model ============
    if not skip_weight:
        try:
            cfg_yaml = glob(str(res_dir / "backup" / "*.yaml"))[0]
            cfg_loader = config_loader.LayerConfigLoader(cfg_yaml)
        except:
            raise ValueError("Config file does not exist")
        if args.not_rand_feat:
            new_feat = data.dd_sim.shape[0] + data.tt_sim.shape[0]
            cfg_loader.config["layers"][0]["args"]["in_features"] = new_feat

        if args.hidden_dim != -1:
            # set_all_in_features
            for l in cfg_loader.config["layers"]:
                l["args"]["in_features"] = args.hidden_dim
                l["args"]["out_features"] = args.hidden_dim

        if args.n_layers > 0:
            if args.n_layers > len(cfg_loader.config["layers"]):
                cfg_loader.config["layers"] += [cfg_loader.config["layers"][-1]] * (args.n_layers - len(cfg_loader.config["layers"]))

            cfg_loader.config["layers"] = cfg_loader.config["layers"][:args.n_layers]

            # if not use pca:
            if "skipnode" in cfg_loader.config["layers"][0] and args.n_layers == 1:
                del cfg_loader.config["layers"][0]["skipnode"]        

        model = models.Model(cfg_loader, dd_sim=data.dd_sim, tt_sim=data.tt_sim).to(args.device)
        model_pt = res_dir / "weight" / f"{k_th}_model.pt"
        if not model_pt.exists():
            raise ValueError("model.pt does not exist")
        model.load_state_dict(torch.load(model_pt))
        model.eval()

        predictor = P.HeteroDotProductPredictor().to(args.device)
        # predictor = P.HeteroMLPPredictor(in_features=args.n_features, out_classes=2).to(args.device)
        predictor_pt = res_dir / "weight" / f"{k_th}_predictor.pt"
        if not predictor_pt.exists():
            raise ValueError("predictor.pt does not exist")
        predictor.load_state_dict(torch.load(predictor_pt))
        predictor.eval()

        ret.model=model
        ret.predictor=predictor
                 

    ret.dt_adj_df = data.dt_adj_df
    ret.dd_sim_df = data.dd_sim_df
    ret.tt_sim_df = data.tt_sim_df

    return ret
