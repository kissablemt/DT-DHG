import numpy as np
import torch
from typing import Union, List
import dgl

TOP_LIST = [0.01, 0.02, 0.05, 0.1]

def graph_to_adj(g, etype):
    return g.adjacency_matrix(etype=etype).to_dense().to(g.device)

def construct_graph_bak(dd_adj: torch.Tensor,
                    tt_adj: torch.Tensor,
                    dt_adj: torch.Tensor,
                    train_edge_index: Union[torch.Tensor, np.ndarray, List],
                    test_edge_index: Union[torch.Tensor, np.ndarray, List],
                    device="cpu"):

    # u: drug,  v: target
    u, v = torch.where(dt_adj)

    # choose u,v by index
    test_u, test_v = u[test_edge_index], v[test_edge_index]
    train_u, train_v = u[train_edge_index], v[train_edge_index]

    # DT adj split to train and test
    train_dt_adj = torch.zeros_like(dt_adj)
    train_dt_adj[train_u, train_v] = 1
    test_dt_adj = torch.zeros_like(dt_adj)
    test_dt_adj[test_u, test_v] = 1

    dd_edges = torch.where(dd_adj)
    tt_edges = torch.where(tt_adj)

    g = dgl.heterograph({
        ('drug', 'dd', 'drug'): dd_edges,
        ('target', 'tt', 'target'): tt_edges,
        ('drug', 'dt', 'target'): (u, v),
        ('target', 'td', 'drug'): (v, u),
    }, num_nodes_dict={"drug": dt_adj.shape[0], "target": dt_adj.shape[1]}).to(device)

    train_g = dgl.heterograph({
        ('drug', 'dd', 'drug'): dd_edges,
        ('target', 'tt', 'target'): tt_edges,
        ('drug', 'dt', 'target'): (train_u, train_v),
        ('target', 'td', 'drug'): (train_v, train_u),
    }, num_nodes_dict={"drug": train_dt_adj.shape[0], "target": train_dt_adj.shape[1]}).to(device)

    test_g = dgl.heterograph({
        ('drug', 'dd', 'drug'): dd_edges,
        ('target', 'tt', 'target'): tt_edges,
        ('drug', 'dt', 'target'): (test_u, test_v),
        ('target', 'td', 'drug'): (test_v, test_u),
    }, num_nodes_dict={"drug": test_dt_adj.shape[0], "target": test_dt_adj.shape[1]}).to(device)

    return train_g, test_g, g


def construct_graph(dd_adj: torch.Tensor,
                    tt_adj: torch.Tensor,
                    dt_adj: torch.Tensor,
                    train_edge_index: Union[torch.Tensor, np.ndarray, List],
                    valid_edge_index: Union[torch.Tensor, np.ndarray, List],
                    test_edge_index: Union[torch.Tensor, np.ndarray, List],
                    device="cpu"):

    # u: drug,  v: target
    u, v = torch.where(dt_adj)

    # choose u,v by index
    test_u, test_v = u[test_edge_index], v[test_edge_index]
    train_u, train_v = u[train_edge_index], v[train_edge_index]
    valid_u, valid_v = u[valid_edge_index], v[valid_edge_index]

    # DT adj split to train and test
    train_dt_adj = torch.zeros_like(dt_adj)
    train_dt_adj[train_u, train_v] = 1
    test_dt_adj = torch.zeros_like(dt_adj)
    test_dt_adj[test_u, test_v] = 1
    valid_dt_adj = torch.zeros_like(dt_adj)
    valid_dt_adj[valid_u, valid_v] = 1

    dd_edges = torch.where(dd_adj)
    tt_edges = torch.where(tt_adj)

    g = dgl.heterograph({
        ('drug', 'dd', 'drug'): dd_edges,
        ('target', 'tt', 'target'): tt_edges,
        ('drug', 'dt', 'target'): (u, v),
        ('target', 'td', 'drug'): (v, u),
    }, num_nodes_dict={"drug": dt_adj.shape[0], "target": dt_adj.shape[1]}).to(device)

    train_g = dgl.heterograph({
        ('drug', 'dd', 'drug'): dd_edges,
        ('target', 'tt', 'target'): tt_edges,
        ('drug', 'dt', 'target'): (train_u, train_v),
        ('target', 'td', 'drug'): (train_v, train_u),
    }, num_nodes_dict={"drug": train_dt_adj.shape[0], "target": train_dt_adj.shape[1]}).to(device)

    valid_g = dgl.heterograph({
        ('drug', 'dd', 'drug'): dd_edges,
        ('target', 'tt', 'target'): tt_edges,
        ('drug', 'dt', 'target'): (valid_u, valid_v),
        ('target', 'td', 'drug'): (valid_v, valid_u),
    }, num_nodes_dict={"drug": valid_dt_adj.shape[0], "target": valid_dt_adj.shape[1]}).to(device)


    test_g = dgl.heterograph({
        ('drug', 'dd', 'drug'): dd_edges,
        ('target', 'tt', 'target'): tt_edges,
        ('drug', 'dt', 'target'): (test_u, test_v),
        ('target', 'td', 'drug'): (test_v, test_u),
    }, num_nodes_dict={"drug": test_dt_adj.shape[0], "target": test_dt_adj.shape[1]}).to(device)

    return train_g, valid_g, test_g, g



def construct_multi_graphs(dd_adj: torch.Tensor,
                           tt_adj: torch.Tensor,
                           dt_adj: torch.Tensor,
                           train_edge_index: Union[torch.Tensor, np.ndarray, List],
                           test_edge_index: Union[torch.Tensor, np.ndarray, List],
                           dd_sim: torch.Tensor,
                           tt_sim: torch.Tensor,
                           tops: List = TOP_LIST,
                           device="cpu"):

    train_g, test_g, g = construct_graph(dd_adj=dd_adj, tt_adj=tt_adj, dt_adj=dt_adj, 
                                         train_edge_index=train_edge_index, test_edge_index=test_edge_index, 
                                         device=device)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ tops @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    train_gs = []
    for top in tops:
        _g = dgl.heterograph({
            ('drug', 'dd', 'drug'): torch.where(dd_sim > torch.quantile(dd_sim.ravel(), q=1 - top)),
            ('target', 'tt', 'target'): torch.where(tt_sim > torch.quantile(tt_sim.ravel(), q=1 - top)),
            ('drug', 'dt', 'target'): train_g.edges(etype="dt"),
            ('target', 'td', 'drug'): train_g.edges(etype="td"),
        }, num_nodes_dict={"drug": dd_sim.shape[0], "target": tt_sim.shape[0]}).to(device)
        train_gs.append(_g)

    test_gs = []
    for top in tops:
        _g = dgl.heterograph({
            ('drug', 'dd', 'drug'): torch.where(dd_sim > torch.quantile(dd_sim.ravel(), q=1 - top)),
            ('target', 'tt', 'target'): torch.where(tt_sim > torch.quantile(tt_sim.ravel(), q=1 - top)),
            ('drug', 'dt', 'target'): test_g.edges(etype="dt"),
            ('target', 'td', 'drug'): test_g.edges(etype="td"),
        }, num_nodes_dict={"drug": dd_sim.shape[0], "target": tt_sim.shape[0]}).to(device)
        test_gs.append(_g)
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    return train_gs, test_gs, g
