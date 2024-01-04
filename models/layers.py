import torch
import torch.nn as nn
import dgl.nn as dglnn
from utils import deeper_gcn, base

'''
x, h, z in forward are all dict of {"drug": tensor, "target": tensor}
'''

class HETSAGEConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, aggregator_type: str, batch_norm: bool=False) -> None:
        super().__init__()

        dt_conv = dglnn.SAGEConv(
            in_feats=in_features, out_feats=out_features, aggregator_type=aggregator_type)

        self.conv = dglnn.HeteroGraphConv({
            "dd": dglnn.SAGEConv(in_feats=in_features, out_feats=out_features, aggregator_type=aggregator_type),
            "tt": dglnn.SAGEConv(in_feats=in_features, out_feats=out_features, aggregator_type=aggregator_type),
            "dt": dt_conv,
            "td": dt_conv,
        }, aggregate='sum')

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, g, x, global_bn=None):
        h = self.conv(g, x) # shape is {"drug": (N, 1, out_features), "target": (N, 1, out_features)}, because num_heads=1
        h = {k: v.squeeze(1) for k, v in h.items()} # shape is {"drug": (N, out_features), "target": (N, out_features)}
        if self.batch_norm:
            if global_bn is not None:
                self.bn = global_bn                
            h = {k: self.bn(v) for k, v in h.items()}
        return h

class HETGraphConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, batch_norm: bool=False) -> None:
        super().__init__()

        dt_conv = dglnn.GraphConv(
            in_feats=in_features, out_feats=out_features)

        self.conv = dglnn.HeteroGraphConv({
            "dd": dglnn.GraphConv(in_feats=in_features, out_feats=out_features),
            "tt": dglnn.GraphConv(in_feats=in_features, out_feats=out_features),
            "dt": dt_conv,
            "td": dt_conv,
        }, aggregate='sum')

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, g, x, global_bn=None):
        h = self.conv(g, x) # shape is {"drug": (N, 1, out_features), "target": (N, 1, out_features)}, because num_heads=1
        h = {k: v.squeeze(1) for k, v in h.items()} # shape is {"drug": (N, out_features), "target": (N, out_features)}

        if self.batch_norm:
            if global_bn is not None:
                self.bn = global_bn                
            h = {k: self.bn(v) for k, v in h.items()}
        return h

class HETA(nn.Module):
    def __init__(self, in_features: int, out_features: int, gat_feat_drop: int, gat_attn_drop: int, batch_norm: bool=False) -> None:
        super().__init__()

        dt_gatconv = dglnn.GATConv(
            in_feats=in_features, out_feats=out_features, num_heads=1, feat_drop=gat_feat_drop, attn_drop=gat_attn_drop)

        self.conv = dglnn.HeteroGraphConv({
            "dd": dglnn.GATConv(in_feats=in_features, out_feats=out_features, num_heads=1, feat_drop=gat_feat_drop, attn_drop=gat_attn_drop),
            "tt": dglnn.GATConv(in_feats=in_features, out_feats=out_features, num_heads=1, feat_drop=gat_feat_drop, attn_drop=gat_attn_drop),
            "dt": dt_gatconv,
            "td": dt_gatconv,
        }, aggregate='sum')

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, g, x, global_bn=None):
        h = self.conv(g, x) # shape is {"drug": (N, 1, out_features), "target": (N, 1, out_features)}, because num_heads=1
        h = {k: v.squeeze(1) for k, v in h.items()} # shape is {"drug": (N, out_features), "target": (N, out_features)}
        if self.batch_norm:
            if global_bn is not None:
                self.bn = global_bn                
            h = {k: self.bn(v) for k, v in h.items()}
        return h

class HETAv2(nn.Module):
    def __init__(self, in_features: int, out_features: int, gat_feat_drop: int, gat_attn_drop: int, batch_norm: bool=False) -> None:
        super().__init__()

        dt_gatconv = dglnn.GATv2Conv(
            in_feats=in_features, out_feats=out_features, num_heads=1, feat_drop=gat_feat_drop, attn_drop=gat_attn_drop)

        self.conv = dglnn.HeteroGraphConv({
            "dd": dglnn.GATv2Conv(in_feats=in_features, out_feats=out_features, num_heads=1, feat_drop=gat_feat_drop, attn_drop=gat_attn_drop),
            "tt": dglnn.GATv2Conv(in_feats=in_features, out_feats=out_features, num_heads=1, feat_drop=gat_feat_drop, attn_drop=gat_attn_drop),
            "dt": dt_gatconv,
            "td": dt_gatconv,
        }, aggregate='sum')

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_features)

    def forward(self, g, x, global_bn=None):
        h = self.conv(g, x) # shape is {"drug": (N, 1, out_features), "target": (N, 1, out_features)}, because num_heads=1
        h = {k: v.squeeze(1) for k, v in h.items()} # shape is {"drug": (N, out_features), "target": (N, out_features)}
        if self.batch_norm:
            if global_bn is not None:
                self.bn = global_bn                
            h = {k: self.bn(v) for k, v in h.items()}
        return h
        

class SkipNode(nn.Module):
    def __init__(self, conv_layer, skip_rate: float, skip_type: str, enable_in_test: bool) -> None:
        super().__init__()
        self.conv_layer = conv_layer
        self.skip_rate = skip_rate
        self.skip_type = skip_type
        self.enable_in_test = enable_in_test
    
    def forward(self, g, x, global_bn=None):
        h = self.conv_layer(g, x, global_bn)

        if not self.training and not self.enable_in_test: 
            return h
        
        z = {
            "drug": deeper_gcn.skip_node(x_old=x["drug"], x_new=h["drug"], 
                                         skip_rate=self.skip_rate, skip_type=self.skip_type, degree=g.in_degrees(etype="dd"), 
                                         device=g.device),
            "target": deeper_gcn.skip_node(x_old=x["target"], x_new=h["target"], 
                                           skip_rate=self.skip_rate, skip_type=self.skip_type, degree=g.in_degrees(etype="tt"), 
                                           device=g.device),
        }
        return z


class SkipNodeG(nn.Module):
    def __init__(self, conv_layer, skip_rate: float, skip_type: str, enable_in_test: bool) -> None:
        super().__init__()
        self.conv_layer = conv_layer
        self.skip_rate = skip_rate
        self.skip_type = skip_type
        self.enable_in_test = enable_in_test
    
    def forward(self, g, x, global_bn=None):
        h = self.conv_layer(g, x, global_bn)

        if not self.training and not self.enable_in_test: 
            return h
        
        z = {
            "drug": deeper_gcn.skip_node_g(x_old=x["drug"], x_new=h["drug"], 
                                         skip_rate=self.skip_rate, skip_type=self.skip_type, adj=g.adjacency_matrix(etype="dd").to_dense().to(g.device)),
            "target": deeper_gcn.skip_node_g(x_old=x["target"], x_new=h["target"], 
                                           skip_rate=self.skip_rate, skip_type=self.skip_type, adj=g.adjacency_matrix(etype="tt").to_dense().to(g.device)),
        }
        return z
    

class RankBern(nn.Module):
    def __init__(self, conv_layer, dd_sim, tt_sim, p_list=None, v_list=None, enable_in_test: bool=False) -> None:
        super().__init__()
        self.conv_layer = conv_layer
        self.enable_in_test = enable_in_test

        if p_list is None:
            p_list = [  95,   90,   85,   80,   75]
        if v_list is None:
            v_list = [1.00, 0.60, 0.30, 0.15, 0.10]

        self.dd_sim = base.rank_vec(dd_sim, p_list, v_list)
        self.tt_sim = base.rank_vec(tt_sim, p_list, v_list)

    def forward(self, g, x, global_bn=None):
        if not self.training and not self.enable_in_test: 
            gn = g
        else:
            gn = deeper_gcn.bern_edge(graph=g, dd_sim=self.dd_sim, tt_sim=self.tt_sim)
        
        h = self.conv_layer(gn, x, global_bn)
        return h


class RankBernGL(nn.Module):
    def __init__(self, conv_layer, dd_sim, tt_sim, enable_in_test: bool=False) -> None:
        super().__init__()
        self.conv_layer = conv_layer
        self.enable_in_test = enable_in_test

        self.dd_sim = self.GL(dd_sim)
        self.tt_sim = self.GL(tt_sim)

    def GL(self, sim):
        def G(x):
            z = 25 / x ** 2
            return torch.where(z > 1, 1, z)

        def L(x):
            z = 1 / x ** 2
            return torch.where(z > 1, 1, z)
        
        R_G = base.global_rank(sim)
        R_L = base.local_rank(sim)

        S_G = G(R_G)
        S_L = L(R_L)

        return torch.max(S_G, S_L)
    

    def forward(self, g, x, global_bn=None):
        if not self.training and not self.enable_in_test: 
            gn = g
        else:
            gn = deeper_gcn.bern_edge(graph=g, dd_sim=self.dd_sim, tt_sim=self.tt_sim)
        
        h = self.conv_layer(gn, x, global_bn)
        return h


class RankBernR(nn.Module):
    def __init__(self, conv_layer, dd_sim, tt_sim, p_list=None, v_list=None, sample_rate: float=0.5, enable_in_test: bool=False) -> None:
        super().__init__()
        self.conv_layer = conv_layer
        self.enable_in_test = enable_in_test
        self.sample_rate = sample_rate

        if p_list is None:
            p_list = [  95,   90,   85,   80,   75]
        if v_list is None:
            v_list = [1.00, 0.60, 0.30, 0.15, 0.10]

        self.dd_sim = base.rank_vec(dd_sim, p_list, v_list)
        self.tt_sim = base.rank_vec(tt_sim, p_list, v_list)

    def forward(self, g, x, global_bn=None):
        if not self.training and not self.enable_in_test: 
            gn = g
        else:
            gn = deeper_gcn.bern_edge_rate(graph=g, dd_sim=self.dd_sim, tt_sim=self.tt_sim, sample_rate=self.sample_rate)
        
        h = self.conv_layer(gn, x, global_bn)
        return h