import torch
from torch.distributions import Bernoulli
from torch.utils.data import DataLoader, WeightedRandomSampler
import dgl
from dgl import DGLHeteroGraph


def bern_edge(graph, dd_sim, tt_sim):
    dd_adj = dd_sim.bernoulli().int()
    tt_adj = tt_sim.bernoulli().int()

    return dgl.heterograph({
        ('drug', 'dd', 'drug'): torch.where(dd_adj),
        ('target', 'tt', 'target'): torch.where(tt_adj),
        ('drug', 'dt', 'target'): graph.edges(etype='dt'),
        ('target', 'td', 'drug'): graph.edges(etype='td'),
    }).to(device=graph.device)

def bern_edge_rate(graph, dd_sim, tt_sim, sample_rate=0.5):
    old_dd_adj = graph.adjacency_matrix(etype="dd").to_dense().int().to(graph.device)
    old_tt_adj = graph.adjacency_matrix(etype="tt").to_dense().int().to(graph.device)

    new_dd_adj = dd_sim.bernoulli().int()
    new_tt_adj = tt_sim.bernoulli().int()

    if sample_rate == 0.0: 
        dd_adj = old_dd_adj
        tt_adj = old_tt_adj
    else:
        dd_adj = skip_node(x_old=old_dd_adj, x_new=new_dd_adj, skip_rate=sample_rate, skip_type="uniform", degree=graph.in_degrees(etype="dd"), device=graph.device)
        tt_adj = skip_node(x_old=old_tt_adj, x_new=new_tt_adj, skip_rate=sample_rate, skip_type="uniform", degree=graph.in_degrees(etype="tt"), device=graph.device)
    
    return dgl.heterograph({
        ('drug', 'dd', 'drug'): torch.where(dd_adj),
        ('target', 'tt', 'target'): torch.where(tt_adj),
        ('drug', 'dt', 'target'): graph.edges(etype='dt'),
        ('target', 'td', 'drug'): graph.edges(etype='td'),
    }).to(device=graph.device)

def drop_edge(graph: DGLHeteroGraph, p=0.9):
    graph = graph.clone()
    dist = Bernoulli(p)
    for etype in ["dd", "tt"]:
        samples = dist.sample(torch.Size([graph.num_edges(etype)]))
        eids_to_remove = graph.edges(form='eid', etype=etype)[samples.bool().to(graph.device)]
        graph.remove_edges(eids_to_remove, etype=etype)
    return graph
    

def skip_node_mask(N, skip_rate=0.5, skip_type="uniform", degree=None, device="cpu"):
    # from https://github.com/WeigangLu/SkipNode/blob/SkipNode/model.py
    mask = torch.FloatTensor([1.0 for _ in range(N)])

    if skip_type == 'degree':
        prob = degree / (degree.sum() + 1e-8)
    else:
        prob = torch.ones(N) / N

    index = torch.Tensor([i for i in range(N)]).to(device)
    size = int(N * skip_rate)
    dataloader = DataLoader(dataset=index, batch_size=size,
                            sampler=WeightedRandomSampler(prob, size, replacement=False))
    sampled_idx = None
    for data in dataloader:
        sampled_idx = data
    sampled_idx = sampled_idx.to(torch.int64).cpu()
    mask = mask.index_fill_(0, sampled_idx, 0)
    mask = mask.unsqueeze(1).to(device)
    return mask


def skip_node(x_old, x_new, skip_rate=0.5, skip_type="uniform", degree=None, device="cpu"):
    # from https://github.com/WeigangLu/SkipNode/blob/SkipNode/model.py
    mask = skip_node_mask(x_old.shape[0], skip_rate=skip_rate, skip_type=skip_type, degree=degree, device=device)
    out = mask * (x_new - x_old) + x_old # GitHub 实现
    # out = mask * x_new + (1 - mask) * x_old # 论文中的公式，实测效果差不多，但GitHub实现快
    return out


def skip_node_g_mask(N, skip_rate=0.5, skip_type="uniform", degree=None, device="cpu"):
    # from https://github.com/WeigangLu/SkipNode/blob/SkipNode/model.py

    if skip_type == 'degree':
        prob = degree / (degree.sum() + 1e-8)
    else:
        prob = torch.ones(N) * skip_rate
    
    return prob.bernoulli().unsqueeze(1).to(device)

def skip_node_g(x_old, x_new, adj, skip_rate=0.5, skip_type="uniform"):
    # from https://github.com/WeigangLu/SkipNode/blob/SkipNode/model.py

    N = x_old.shape[0]
    if skip_type == "uniform":
        prob = torch.ones(N) * skip_rate
        mask = prob.bernoulli().unsqueeze(1).to(adj.device)
    else:
        deg = adj.sum(dim=1)
        prob = 1 - deg / (adj @ deg)
        mask = prob.bernoulli().unsqueeze(1).to(adj.device)
    
    out = mask * (x_new - x_old) + x_old # GitHub 实现
    # out = mask * x_new + (1 - mask) * x_old # 论文中的公式，实测效果差不多，但GitHub实现快
    return out