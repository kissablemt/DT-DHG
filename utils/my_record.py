import json
from pathlib import Path
import shutil
from easydict import EasyDict
import copy
import numpy as np
import torch
from dgl.data.utils import save_graphs, load_graphs

from utils import base


class MyRecord:
    def __init__(self) -> None:
        self.train = EasyDict(loss=[], auc=[], aupr=[], hit_at_10=[])
        self.valid = EasyDict(loss=[], auc=[], aupr=[], hit_at_10=[])
        self.best = EasyDict(
            loss = {"train": 0, "test": 0, "epoch": 0, "model": None},
            auc = {"train": 0, "test": 0, "epoch": 0, "model": None},
            aupr = {"train": 0, "test": 0, "epoch": 0, "model": None},
            hit_at_10 = {"train": 0, "test": 0, "epoch": 0, "model": None},
        )
        
    def update(self, res: EasyDict, epoch: int, model, save_best_by: str = "auc", predictor=None):
        self.train.loss.append(res.train.loss)
        self.valid.loss.append(res.valid.loss)

        for metric in ["auc", "aupr", "hit_at_10", "loss"]:
            if res.train.get(metric, None) and res.valid.get(metric, None):
                if metric != "loss":
                    self.train[metric].append(res.train[metric])
                    self.valid[metric].append(res.valid[metric])
                if save_best_by == metric:
                    if self.best[metric].test < res.valid[metric]:
                        self.best[metric].test = res.valid[metric]
                        self.best[metric].train = res.train[metric]
                        self.best[metric].epoch = epoch
                        self.best[metric].model = copy.deepcopy(copy.deepcopy(model.state_dict()))
                        if predictor is not None:
                            self.best[metric].predictor = copy.deepcopy(copy.deepcopy(predictor.state_dict()))


    def save_result(self, k_th, args, result, task_dir, main_path, data, train_g, valid_g, test_g, train_neg_g, valid_neg_g, test_neg_g, g):
        rec = self

        if isinstance(task_dir, str):
            task_dir = Path(task_dir)
        task_dir.mkdir(parents=True, exist_ok=True)

        # ============ backup ============
        bak_dir = task_dir / "backup"
        bak_dir.mkdir(parents=True, exist_ok=True)

        main_path = Path(main_path)
        main_bak_path = bak_dir / main_path.name
        shutil.copy(main_path, main_bak_path)

        config_path = Path(args.config)
        config_bak_path = bak_dir / config_path.name
        shutil.copy(config_path, config_bak_path)

        # ============ save args ============
        args_path = task_dir / "args.json"
        with open(args_path, "w+") as f:
            f.write(json.dumps(args.__dict__, ensure_ascii=False, default=base.default_dump, indent=4))
            

        # ============ save result ===========
        krec_dir = task_dir / "record"
        krec_dir.mkdir(parents=True, exist_ok=True)

        krec_path = krec_dir / f"{k_th}.json"
        new_rec = copy.deepcopy(rec)
        for k in new_rec.best.keys():
            new_rec.best[k].model = ""
            new_rec.best[k].predictor = ""
        with open(krec_path, "w+") as f:
            f.write(json.dumps(new_rec.__dict__, ensure_ascii=False, default=base.default_dump))


        kfold_dir = task_dir / "result"
        kfold_dir.mkdir(parents=True, exist_ok=True)

        pos_score = result["pos_score"]
        neg_score = result["neg_score"]
        scores = torch.cat([pos_score.detach(), neg_score.detach()]).cpu().numpy()
        labels = torch.cat(
            [torch.ones(pos_score.cpu().shape[0]), torch.zeros(neg_score.cpu().shape[0])]
        ).numpy()

        np.save(kfold_dir / f"{k_th}_scores.npy", scores)
        np.save(kfold_dir / f"{k_th}_labels.npy", labels)

        drug_target_features = result["h"] 
        torch.save(drug_target_features["drug"].cpu(), kfold_dir / f"{k_th}_drug_h.pt")
        torch.save(drug_target_features["target"].cpu(), kfold_dir / f"{k_th}_target_h.pt")

        node_features = data["node_features"]
        torch.save(node_features["drug"].cpu(), kfold_dir / f"drug_init.pt")
        torch.save(node_features["target"].cpu(), kfold_dir / f"target_init.pt")


        # test_g = result["test_g"]
        # torch.save(test_g.adjacency_matrix(etype="dt").to_dense().cpu(), kfold_dir / f"{k_th}_test_adj.pt")

        # test_neg_g = result["test_neg_g"]
        # torch.save(test_neg_g.adjacency_matrix(etype="dt").to_dense().cpu(), kfold_dir / f"{k_th}_test_neg_adj.pt")

        # ================== save graphs ==================
        graph_dir = task_dir / "graph" 
        graph_dir.mkdir(parents=True, exist_ok=True)
        save_graphs(str(graph_dir / f"{k_th}_train_g.bin"), train_g)
        save_graphs(str(graph_dir / f"{k_th}_test_g.bin"), test_g)
        save_graphs(str(graph_dir / f"{k_th}_valid_g.bin"), valid_g)
        save_graphs(str(graph_dir / f"{k_th}_train_neg_g.bin"), train_neg_g)
        save_graphs(str(graph_dir / f"{k_th}_test_neg_g.bin"), test_neg_g)
        save_graphs(str(graph_dir / f"{k_th}_valid_neg_g.bin"), valid_neg_g)
        save_graphs(str(graph_dir / "g.bin"), g)


        # ============ save best ============
        if args.save_best:
            weight_dir = task_dir / "weight"
            weight_dir.mkdir(parents=True, exist_ok=True)

            best_one = rec.best.get(args.save_best_by)

            filename = weight_dir / f"{k_th}_model.pt"
            torch.save(best_one.model, filename)

            filename = weight_dir / f"{k_th}_predictor.pt"
            torch.save(best_one.predictor, filename)

