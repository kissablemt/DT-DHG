import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
import dgl.function as fn


class HeteroDotProductPredictor(nn.Module):
    def __init__(self, use_sigmod=False):
        super().__init__()
        self.use_sigmod = use_sigmod

    # https://docs.dgl.ai/guide/training-edge.html#guide-training-edge-classification-heterogeneous-graph
    def forward(self, graph, h, etype):
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            if self.use_sigmod:
                return torch.sigmoid(graph.edges[etype].data['score'])
            return graph.edges[etype].data['score']
        

class HeteroMLPPredictor(nn.Module):
    # https://docs.dgl.ai/guide/training-edge.html#guide-training-edge-classification-heterogeneous-graph
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        h_u = edges.src['h']
        h_v = edges.dst['h']
        score = self.W(torch.cat([h_u, h_v], 1))
        return {'score': score}

    def forward(self, graph, h, etype):
        # h contains the node representations for each edge type computed from
        # the GNN for heterogeneous graphs defined in the node classification
        # section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
            graph.apply_edges(self.apply_edges, etype=etype)
            return graph.edges[etype].data['score'].softmax(dim=1)[:,1].view(-1, 1)