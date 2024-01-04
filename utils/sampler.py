import dgl
from dgl import DGLHeteroGraph
import torch


def dt_negative_graph_global(graph: DGLHeteroGraph, k: int) -> DGLHeteroGraph:
    """
    Generate a heterogeneous graph with negative samples using global uniform negative sampling.

    Args:
        graph (DGLHeteroGraph): The input heterogeneous graph.
        k (int): The number of negative samples per edge.

    Returns:
        DGLHeteroGraph: The output heterogeneous graph with negative samples.
    """

    src_type, edge_type, dst_type = ("drug", "dt", "target") 

    neg_src, neg_dst = dgl.sampling.global_uniform_negative_sampling(
        g=graph, 
        num_samples=int(k * graph.num_edges(etype=edge_type)), 
        replace=True, 
        etype=(src_type, edge_type, dst_type)
    )

    return dgl.heterograph({
        ('drug', 'dd', 'drug'): graph.edges(etype="dd"),
        ('target', 'tt', 'target'): graph.edges(etype="tt"),
        ('drug', 'dt', 'target'): (neg_src, neg_dst),
        ('target', 'td', 'drug'): (neg_dst, neg_src),
    }, num_nodes_dict={"drug": graph.num_nodes(src_type), "target": graph.num_nodes(dst_type)}, idtype=torch.int32)


def dt_negative_graph(graph: DGLHeteroGraph, k: int, drop_known=True) -> DGLHeteroGraph:
    """https://docs.dgl.ai/guide/training-link.html#guide-training-link-prediction

    Training a link prediction model involves comparing the scores between nodes 
    connected by an edge against the scores between an arbitrary pair of nodes. 
    For example, given an edge connecting u and v, 
    we encourage the score between node u and v to be higher than the score between node u
    and a sampled node v′ from an arbitrary noise distribution v′∼Pn(v). 
    Such methodology is called negative sampling.

    Args:
        graph (DGLHeteroGraph): The input heterogeneous graph.
        k (int): The number of negative samples per edge.
        drop_known (bool, optional): Whether to drop the known edges. Defaults to True.

    Returns:
        DGLHeteroGraph: The output heterogeneous graph with negative samples.
        
    """
    device = graph.device
    
    src_type, edge_type, dst_type = ("drug", "dt", "target")
    
    src, dst = graph.edges(etype=edge_type)
    neg_dst = dst.repeat_interleave(k)
    neg_src = torch.randint(0, graph.num_nodes(src_type), (len(dst) * k,)).to(device)
    
    if drop_known:
        neg_adj = torch.zeros((graph.num_nodes(src_type), graph.num_nodes(dst_type))).to(device)
        neg_adj[neg_src, neg_dst] = 1
        neg_adj[graph.edges(etype=edge_type)] = 0

        return dgl.heterograph({
            ('drug', 'dd', 'drug'): graph.edges(etype="dd"),
            ('target', 'tt', 'target'): graph.edges(etype="tt"),
            ('drug', 'dt', 'target'): torch.where(neg_adj),
            ('target', 'td', 'drug'): torch.where(neg_adj.T),
        }, num_nodes_dict={"drug": neg_adj.shape[0], "target": neg_adj.shape[1]}, idtype=torch.int32)
    else:
        return dgl.heterograph({
            ('drug', 'dd', 'drug'): graph.edges(etype="dd"),
            ('target', 'tt', 'target'): graph.edges(etype="tt"),
            ('drug', 'dt', 'target'): (neg_src, neg_dst),
            ('target', 'td', 'drug'): (neg_dst, neg_src),
        }, num_nodes_dict={"drug": graph.num_nodes(src_type), "target": graph.num_nodes(dst_type)}, idtype=torch.int32)  