from sklearn.decomposition import PCA
import torch
import pandas as pd
from sklearn.model_selection import KFold
from easydict import EasyDict as edict
from pathlib import Path


def preparation(n_features, dataset_dir, top_thr, not_rand_feat=True, pca=False, device="cpu"):
    if isinstance(dataset_dir, str):
        dataset_dir = Path(dataset_dir)

    top_thr_per = "{:.1%}".format(top_thr) # for example, "5.0%"

    # ============ load dataset in Pandas ============
    dd_sim_df = pd.read_csv(dataset_dir /
                            "adj" / "sim" / "dd.csv", index_col=0)
    tt_sim_df = pd.read_csv(dataset_dir /
                            "adj" / "sim" / "tt.csv", index_col=0)

    dd_adj_df = pd.read_csv(dataset_dir /
                            "adj" / top_thr_per / "dd.csv", index_col=0)
    tt_adj_df = pd.read_csv(dataset_dir /
                            "adj" / top_thr_per / "tt.csv", index_col=0)
    dt_adj_df = pd.read_csv(dataset_dir /
                            "adj" / top_thr_per / "dt.csv", index_col=0)

    # ============ Pandas to Tensor ============
    # Similarity Matrix
    dd_sim = torch.from_numpy(dd_sim_df.values).to(device)
    tt_sim = torch.from_numpy(tt_sim_df.values).to(device)

    # Adjacency Matrix
    dd_adj = torch.from_numpy(dd_adj_df.values).to(device)
    tt_adj = torch.from_numpy(tt_adj_df.values).to(device)
    dt_adj = torch.from_numpy(dt_adj_df.values).to(device)

    # ============ Create features for nodes ============
    if not_rand_feat:
        node_features = {
            "drug": torch.cat([dd_sim, torch.zeros(dd_sim.shape[0], tt_sim.shape[0]).to(device)], dim=1).float(),
            "target": torch.cat([torch.zeros(tt_sim.shape[0], dd_sim.shape[0]).to(device), tt_sim], dim=1).float(),
        }
        if pca:
            # 按照给出的示例合并特征矩阵
            drug_features = torch.cat([dd_sim, torch.zeros(dd_sim.shape[0], tt_sim.shape[0]).to(device)], dim=1).float()
            target_features = torch.cat([torch.zeros(tt_sim.shape[0], dd_sim.shape[0]).to(device), tt_sim], dim=1).float()

            # 创建PCA对象，设置n_components为32
            feat_pca = PCA(n_components=n_features)

            # 使用numpy()将特征矩阵转换为NumPy数组
            drug_features_np = drug_features.cpu().numpy()
            target_features_np = target_features.cpu().numpy()

            # 对药物特征矩阵和靶点特征矩阵分别进行PCA降维
            feat_pca.fit(drug_features_np)
            feat_pca.fit(target_features_np)

            # 使用PCA对象将特征降维
            compressed_drug_features = feat_pca.transform(drug_features_np)
            compressed_target_features = feat_pca.transform(target_features_np)

            node_features = {
                "drug": torch.from_numpy(compressed_drug_features).to(device),
                "target": torch.from_numpy(compressed_target_features).to(device),
            }
            
    else:
        node_features = {
            "drug": torch.randn(dd_sim.shape[0], n_features).to(device),
            "target": torch.randn(tt_sim.shape[0], n_features).to(device),
        }

    return edict(dd_sim=dd_sim, tt_sim=tt_sim,
                 dd_adj=dd_adj, tt_adj=tt_adj, dt_adj=dt_adj,
                 node_features=node_features, dd_sim_df=dd_sim_df, tt_sim_df=tt_sim_df, dt_adj_df=dt_adj_df)


def train_test_split(dt_adj, kfold: int, train_ratio: float):
    # Number of 1 in DT adjacency matrix
    n_dti = int(torch.sum(dt_adj))

    # Generate random edge index
    rand_eids = torch.randperm(n_dti)
    
    run_once = kfold == 0
    if run_once:
        test_size = int(n_dti * (1 - train_ratio))
        ready_to_run = [
            (rand_eids[test_size:],
             rand_eids[:test_size],)
        ]
    else:
        kf = KFold(n_splits=kfold)
        kf.get_n_splits(rand_eids)
        ready_to_run = kf.split(rand_eids)
    return ready_to_run

def train_valid_test_split(dt_adj, kfold: int, train_ratio: float, valid_ratio: float):
    """
    return [(train_index, valid_index, test_index), ...]
    # +-----------------+-----------------+-----------------+-----------------+
    # |    kfold > 1    |   train_ratio   |    val_ratio    |    test_ratio   |
    # | --------------- | --------------- | --------------- | --------------- |
    # |      True       |   a  * (1-1/K)  | (1-a) * (1-1/K) |       1/K       |  ignore val_ratio
    # |      False      |        a        |        b        |    1 - a - b    |
    # +-----------------+-----------------+-----------------+-----------------+
    """
    # Number of 1 in DT adjacency matrix
    n_dti = int(torch.sum(dt_adj))

    # Generate random edge index
    rand_eids = torch.randperm(n_dti)
    
    run_once = kfold == 0 or kfold == 1
    if run_once:
        train_size = int(n_dti * train_ratio)
        valid_size = int(n_dti * valid_ratio)
        
        ready_to_run = [
            (rand_eids[:train_size], rand_eids[train_size:train_size+valid_size], rand_eids[train_size+valid_size:])
        ]
    else:
        kf = KFold(n_splits=kfold)
        ready_to_run = []
        for train_valid_set, test_set in kf.split(rand_eids):
            train_size = int(len(train_valid_set) * train_ratio)
            
            ready_to_run.append(
                (rand_eids[train_valid_set[:train_size]], rand_eids[train_valid_set[train_size:]], rand_eids[test_set])
            )

    return ready_to_run