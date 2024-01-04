import copy
import torch
from pathlib import Path
from easydict import EasyDict as edict
import logging
import itertools
import argparse
import time

import models
from models import predictors as P
from models import losses as L
from utils import data_loader, dti_graph, sampler, config_loader, my_record, training_utils, base

 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path("./data")
TASK_DIR = "results/:)" # Will be changed after the argument is parsed


def parse_arguments():
    global TASK_DIR

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default="", help="task name")
    parser.add_argument('--output_dir', type=str, default="results", help="output directory")

    parser.add_argument('--config', type=str, help="config to use")
    parser.add_argument('--dataset', type=str, default="EN", help="dataset to use")
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs to train')
    parser.add_argument('--top_thr', type=float, default=0.05, help='top threshold of similarity matrix')

    parser.add_argument('--n_features', type=int, default=32, help='number of features to use')
    parser.add_argument('--hidden_dim', type=int, default=-1, help='hidden dimension of the model, -1: use config file')
    parser.add_argument('--sim2feat', dest='not_rand_feat', action='store_true')
    parser.add_argument('--pca', dest='pca', action='store_true')

    parser.add_argument('--n_layers', type=int, default=0,)
    parser.add_argument('--negative_sample', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    # +-----------------+-----------------+-----------------+-----------------+
    # |    kfold > 1    |   train_ratio   |    val_ratio    |    test_ratio   |
    # | --------------- | --------------- | --------------- | --------------- |
    # |      True       |   a  * (1-1/K)  | (1-a) * (1-1/K) |       1/K       |  ignore val_ratio
    # |      False      |        a        |        b        |    1 - a - b    |
    # +-----------------+-----------------+-----------------+-----------------+
    parser.add_argument('--kfold', type=int, default=0, help='k-fold cross validation')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='train ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='val ratio')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str,
                        default=DEVICE, help='device to use')

    parser.add_argument('--save_best', dest='save_best', action='store_true')
    parser.add_argument('--calc_auc', dest='calc_auc', action='store_true')
    parser.add_argument('--calc_aupr', dest='calc_aupr', action='store_true')
    parser.add_argument('--calc_hit_at_10',
                        dest='calc_hit_at_10', action='store_true')

    parser.add_argument('--save_best_by', type=str, choices=[
                        'auc', 'aupr', 'hit_at_10', 'loss'], default="aupr", help="save best model by which metric")
    
    parser.add_argument('--sigmoid', dest='sigmoid', action='store_true')
    
    parser.add_argument('--tmp', dest='tmp', action='store_true')
    
        
    args = parser.parse_args()
    
    if args.save_best_by == "auc":
        args.calc_auc = True    
    elif args.save_best_by == "aupr":
        args.calc_aupr = True
    elif args.save_best_by == "hit_at_10":
        args.calc_hit_at_10 = True

    handlers=[logging.StreamHandler()]

    res_root_dir = Path(args.output_dir)
    cur_time = time.strftime("%m%d-%H%M%S", time.localtime())

    if args.tmp:
        args.epochs = 1

    else:
        TASK_DIR = res_root_dir / (cur_time + "-" + args.name)
        TASK_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(TASK_DIR / 'output.log'))

    logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers)
    
    return args


def main(args):
    base.setup_seed(seed=args.seed)

    # ============ Preparation ============
    data = data_loader.preparation(
        n_features=args.n_features, dataset_dir=DATA_DIR / args.dataset, 
        top_thr=args.top_thr, not_rand_feat=args.not_rand_feat, pca=args.pca, device=args.device)

    # ============ Train/Valid/Test Split ============
    run_once = args.kfold == 0 or args.kfold == 1
    ready_to_run = data_loader.train_valid_test_split(dt_adj=data.dt_adj, kfold=args.kfold, train_ratio=args.train_ratio, valid_ratio=args.val_ratio)
    # ready_to_run = ready_to_run[2:3]

    # ============ Training ============
    for i, (train_index, valid_index, test_index) in enumerate(ready_to_run):
        if not run_once:
            logging.info(f"Fold {i}:")

        # ============ Model ============
        cfg_loader = config_loader.LayerConfigLoader(args.config)
        if args.hidden_dim != -1:
            # set_all_in_features
            for l in cfg_loader.config["layers"]:
                l["args"]["in_features"] = args.hidden_dim
                l["args"]["out_features"] = args.hidden_dim


        if args.n_layers > 0:
            if args.n_layers > len(cfg_loader.config["layers"]):
                cfg_loader.config["layers"] += [cfg_loader.config["layers"][-1]] * (args.n_layers - len(cfg_loader.config["layers"]))

            cfg_loader.config["layers"] = cfg_loader.config["layers"][:args.n_layers]

        if args.not_rand_feat:
            new_feat = data.node_features["drug"].shape[1]
            cfg_loader.config["layers"][0]["args"]["in_features"] = new_feat

        # if not use pca:
        if "skipnode" in cfg_loader.config["layers"][0]:
            if args.n_layers == 1 or args.not_rand_feat:
                del cfg_loader.config["layers"][0]["skipnode"]
        
        model = models.Model(cfg_loader, dd_sim=data.dd_sim, tt_sim=data.tt_sim).to(args.device)
        # print(model)
        predictor = P.HeteroDotProductPredictor(args.sigmoid).to(args.device)
        losses = L.Loss(args.config)
        optimizer = torch.optim.Adam(itertools.chain(model.parameters(), predictor.parameters()), lr=args.lr)

        # ============ Construct positive graph and negative graph ============
        train_g, valid_g, test_g, g = dti_graph.construct_graph(dd_adj=data.dd_adj, tt_adj=data.tt_adj, dt_adj=data.dt_adj,
                                                       train_edge_index=train_index, valid_edge_index=valid_index, test_edge_index=test_index,
                                                       device=args.device)
        train_neg_g = sampler.dt_negative_graph_global(train_g, k=args.negative_sample)
        valid_neg_g = sampler.dt_negative_graph_global(valid_g, k=args.negative_sample)
        test_neg_g = sampler.dt_negative_graph_global(test_g, k=args.negative_sample)

        # ============ Record ============
        rec = my_record.MyRecord()

        # ============ Training ============
        this_res = training_utils.train_n_epochs(n_epochs=args.epochs, data=data, 
            model=model, predictor=predictor, criterion=losses, optimizer=optimizer,
            g=g, train_g=train_g, train_neg_g=train_neg_g, valid_g=valid_g, valid_neg_g=valid_neg_g, test_g=test_g, test_neg_g=test_neg_g,
            rec=rec, calc_auc=args.calc_auc, calc_aupr=args.calc_aupr, calc_hit_at_10=args.calc_hit_at_10, save_best_by=args.save_best_by
        )

        rec.save_result(k_th=i, result=this_res, args=args, 
                        task_dir=TASK_DIR, main_path=__file__,
                        data=data, 
                        train_g=train_g, valid_g=valid_g,test_g=test_g, 
                        train_neg_g=train_neg_g, valid_neg_g=valid_neg_g, test_neg_g=test_neg_g, g=g)
    

if __name__ == '__main__':
    args = parse_arguments()
    
    # args.config = "configs/inc-bx-a/SAGE.yaml"
    # args.epochs = 1
    # args.tmp = True

    # args.n_features = 32
    # args.hidden_dim = 32
    # args.not_rand_feat = True

    main(args)