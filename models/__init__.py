
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
import dgl.function as fn

from typing import Union
from pathlib import Path

from . import layers, losses
from utils.config_loader import LayerConfigLoader


class Model(nn.Module):
    def __init__(self, config_loader: Union[str, LayerConfigLoader, Path], dd_sim=None, tt_sim=None):
        super().__init__()

        if isinstance(config_loader, LayerConfigLoader):
            self.config = config_loader.config
        else:
            config_loader = LayerConfigLoader(config_loader)
            self.config = config_loader.config

        self.layers = nn.ModuleList()

        self.use_global_bn = self.config.get("global_batch_norm", False)
        if self.use_global_bn:
            self.global_bn = nn.BatchNorm1d(config_loader.get_out_features())


        for layer_config in self.config["layers"]:
            args = layer_config["args"]
            layer = getattr(layers, layer_config["name"])(**args)
            
            if layer_config.get("skipnode", None):
                skipnode_args = layer_config["skipnode"]
                skipnode = getattr(layers, "SkipNode")
                layer = skipnode(layer, **skipnode_args)
            
            if layer_config.get("skipnode_g", None):
                skipnode_g_args = layer_config["skipnode_g"]
                skipnode_g = getattr(layers, "SkipNodeG")
                layer = skipnode_g(layer, **skipnode_g_args)
            
            if layer_config.get("rankbern", None):
                rankbern_args = layer_config["rankbern"]
                rankbern = getattr(layers, "RankBern")
                if None in (dd_sim, tt_sim):
                    raise ValueError("dd_sim and tt_sim must be provided if using RankBern")
                layer = rankbern(layer, dd_sim, tt_sim, **rankbern_args)

            if layer_config.get("rankbern_gl", None):
                rankbern_gl_args = layer_config["rankbern_gl"]
                rankbern_gl = getattr(layers, "RankBernGL")
                if None in (dd_sim, tt_sim):
                    raise ValueError("dd_sim and tt_sim must be provided if using RankBern")
                layer = rankbern_gl(layer, dd_sim, tt_sim, **rankbern_gl_args)
            
            if layer_config.get("rankbern_r", None):
                rankbern_r_args = layer_config["rankbern_r"]
                rankbern_r = getattr(layers, "RankBernR")
                if None in (dd_sim, tt_sim):
                    raise ValueError("dd_sim and tt_sim must be provided if using RankBern")
                layer = rankbern_r(layer, dd_sim, tt_sim, **rankbern_r_args)

            self.layers.append(layer)

        # print(self.layers)

    def forward(self, g, x):
        h = x
        for cfg, layer in zip(self.config["layers"], self.layers):
            h = layer(g, h, self.global_bn if self.use_global_bn else None)
        return h