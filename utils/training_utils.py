import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from easydict import EasyDict as edict
import logging
import itertools
import models
from models import predictors as P
from models import losses as L
from utils import data_loader, dti_graph, sampler, config_loader, my_record, evaluator, base
import copy
from tqdm import tqdm

def test_best(rec: my_record.MyRecord, model, predictor, criterion, g, test_g, test_neg_g, dd_sim, tt_sim, node_features, device="cpu", save_best_by="auc"):
    best_one = rec.best.get(save_best_by)
    model.load_state_dict(best_one.model)

    test_res = train_test(model, method="test",
                          node_features=node_features,
                          graph=test_g, neg_graph=test_neg_g,
                          predictor=predictor, criterion=criterion)

    loss = test_res["loss"]
    all_metrics = evaluator.compute(
        test_res["pos_score"], test_res["neg_score"])

    test_res["test_g"] = test_g
    test_res["test_neg_g"] = test_neg_g

    # auc = evaluator.compute_auc(test_res["pos_score"], test_res["neg_score"])
    # aupr = evaluator.compute_aupr(test_res["pos_score"], test_res["neg_score"])
    hit_at_10 = evaluator.hit_at_k(test_res["score"], test_g.adjacency_matrix(
        etype="dt").to_dense().to(device), k=10)
    # pre_at_10 = evaluator.pre_at_k(test_res["score"], test_g.adjacency_matrix(etype="dt").to_dense().to(device), k=10)

    logging.info("loss %.4f" % loss)
    logging.info("AUC %.4f" % all_metrics["auc"])
    logging.info("AUPR %.4f" % all_metrics["aupr"])
    logging.info("Accuracy %.4f" % all_metrics["accuracy"])
    logging.info("Recall %.4f" % all_metrics["recall"])
    logging.info("Precision %.4f" % all_metrics["precision"])
    logging.info("F1 %.4f" % all_metrics["f1"])
    logging.info("Hit@10 %.4f" % hit_at_10)
    logging.info("Best threshold %.4f" % all_metrics["best_threshold"])
    # logging.info("Pre@10 %.4f" % pre_at_10)

    g_adj = dti_graph.graph_to_adj(g, "dt").int()
    test_adj = dti_graph.graph_to_adj(test_g, "dt").int()
    pos_score = test_res["pos_score"].detach().cpu().flatten()
    neg_score = test_res["neg_score"].detach().cpu().flatten()
    logging.info("Hits@1 %.4f" % evaluator.ogb_hits(pos_score, neg_score, k=1))
    logging.info("Hits@3 %.4f" % evaluator.ogb_hits(pos_score, neg_score, k=3))
    logging.info("Hits@10 %.4f" % evaluator.ogb_hits(pos_score, neg_score, k=10))
    logging.info("Hits@20 %.4f" % evaluator.ogb_hits(pos_score, neg_score, k=20))
    logging.info("Hits@30 %.4f" % evaluator.ogb_hits(pos_score, neg_score, k=30))
    logging.info("Hits@40 %.4f" % evaluator.ogb_hits(pos_score, neg_score, k=40))
    logging.info("Hits@50 %.4f" % evaluator.ogb_hits(pos_score, neg_score, k=50))
    logging.info("MRR %.4f" % evaluator.ogb_mrr(pos_score, neg_score))
    logging.info("Pre@10 %.4f" % evaluator.pre_at_k(test_res["score"], g_adj, k=10))
    logging.info("Pre@20 %.4f" % evaluator.pre_at_k(test_res["score"], g_adj, k=20))
    logging.info("Pre@30 %.4f" % evaluator.pre_at_k(test_res["score"], g_adj, k=30))
    logging.info("Pre@40 %.4f" % evaluator.pre_at_k(test_res["score"], g_adj, k=40))
    logging.info("Pre@50 %.4f" % evaluator.pre_at_k(test_res["score"], g_adj, k=50))

    # print(all_metrics, "\nhit_at_10", hit_at_10)
    return test_res

def train_test(model: nn.Module, predictor: nn.Module, method: str, 
               node_features, graph, neg_graph, 
               criterion: L.Loss,
               optimizer=None):
    if method == "train":
        model.train()
        predictor.train()
        
        # forward
        optimizer.zero_grad()
        h = model(graph, node_features)
        pos_score = predictor(graph, h, etype="dt")
        neg_score = predictor(neg_graph, h, etype="dt")

        local_neg_graph = sampler.dt_negative_graph(graph, k=3, drop_known=False)
        local_neg_score = predictor(local_neg_graph, h, etype="dt")
        loss = criterion(pos_score, neg_score, local_neg_score)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return dict(loss=loss.item(), pos_score=pos_score, neg_score=neg_score, score=h["drug"] @ h["target"].T, h=h)
    else:
        model.eval()
        predictor.eval()
        with torch.no_grad():
            h = model(graph, node_features)
            pos_score = predictor(graph, h, etype="dt")
            neg_score = predictor(neg_graph, h, etype="dt")
            
            local_neg_graph = sampler.dt_negative_graph(graph, k=3, drop_known=False)
            local_neg_score = predictor(local_neg_graph, h, etype="dt")

            loss = criterion(pos_score, neg_score, local_neg_score)
            
            return dict(loss=loss.item(), pos_score=pos_score, neg_score=neg_score, score=h["drug"] @ h["target"].T, h=h)
        
def train_n_epochs(n_epochs: int, data, model: nn.Module, predictor: nn.Module, 
                   criterion: L.Loss, optimizer: torch.optim.Optimizer,
                   g, train_g, train_neg_g, valid_g, valid_neg_g, test_g, test_neg_g,
                   rec: my_record.MyRecord,
                   calc_auc: bool = True, calc_aupr: bool = False, calc_hit_at_10: bool = False, save_best_by="auc"):
    
    device = train_g.device
    for e in tqdm(range(n_epochs)):
        # ======================= train =======================
        train_res = train_test(model, method="train",
                                node_features=data.node_features,
                                graph=train_g, neg_graph=train_neg_g,
                                predictor=predictor, criterion=criterion, optimizer=optimizer)

        # ======================= valid =======================
        valid_res = train_test(model, method="test",
                                node_features=data.node_features,
                                graph=valid_g, neg_graph=valid_neg_g,
                                predictor=predictor, criterion=criterion)

        # ======================= Metrics =======================
        res = edict({
            "train": edict(loss=None, auc=None, aupr=None, hit_at_10=None),
            "valid": edict(loss=None, auc=None, aupr=None, hit_at_10=None),
        })
        res.train.loss = train_res["loss"]
        res.valid.loss = valid_res["loss"]
        
        if calc_auc:
            res.train.auc = evaluator.compute_auc(
                train_res["pos_score"], train_res["neg_score"])
            res.valid.auc = evaluator.compute_auc(
                valid_res["pos_score"], valid_res["neg_score"])

        if calc_aupr:
            res.train.aupr = evaluator.compute_aupr(train_res["pos_score"], train_res["neg_score"])
            res.valid.aupr = evaluator.compute_aupr(valid_res["pos_score"], valid_res["neg_score"])

        if calc_hit_at_10:
            res.train.hit_at_10 = evaluator.ogb_hits(train_res["pos_score"], train_res["neg_score"], k=10)
            res.valid.hit_at_10 = evaluator.ogb_hits(valid_res["pos_score"], valid_res["neg_score"], k=10)

            # res.valid.hit_at_10 = evaluator.pre_at_k(test_res["score"], test_g.adjacency_matrix(etype="dt").to_dense().int().to(device), k=10)

            # res.valid.hit_at_10 = evaluator.hit_at_k(train_res["score"], train_g.adjacency_matrix(etype="dt").to_dense().to(device), k=10)
            # res.valid.hit_at_10 = evaluator.hit_at_k(test_res["score"], test_g.adjacency_matrix(etype="dt").to_dense().to(device), k=10)
        
        rec.update(res=res, epoch=e, model=model, save_best_by=save_best_by, predictor=predictor)

    test_res = test_best(rec, model, predictor=predictor, criterion=criterion, test_g=test_g, test_neg_g=test_neg_g,
                         dd_sim=data.dd_sim, tt_sim=data.tt_sim, node_features=data.node_features, g=g, device=device, save_best_by=save_best_by)
    return copy.deepcopy(test_res)