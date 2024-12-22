from typing import Literal, Optional, Callable

import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn.pytorch import HeteroGraphConv


class HeteroGraphSAGE(nn.Module):

    def __init__(
            self,
            in_feats: dict[str, int],
            hidden_feats: list[int] | int,
            n_layers: Optional[int],
            out_feats: int,
            edge_definitions: set[tuple[str, str, str]],
            activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = F.relu,
            graphsage_aggregator_type: Literal['mean', 'pool', 'lstm', 'gcn']='mean',
            heterogeneous_aggregation: Literal['sum', 'max', 'min', 'mean', 'stack']='mean',
    ) -> None:
        super().__init__()

        if isinstance(hidden_feats, int):
            assert n_layers is not None, "Hidden layers' dimensions were given as a constant but number of layers was not specified."
            hidden_feats = [hidden_feats] * n_layers
        else:
            self._validate_dimensionalities(hidden_feats, n_layers)
        self._validate_all_nodes_have_known_dimensionalities(in_feats, edge_definitions)

        n_layers = len(hidden_feats) + 1
        etypes = {edge_def[1] for edge_def in edge_definitions}
        etype_source_dim = {etype: (in_feats[src_type], in_feats[dst_type]) for src_type, etype, dst_type in edge_definitions}

        self._layers = nn.ModuleList([
            HeteroGraphConv({
                etype: dglnn.SAGEConv(feature_dims, hidden_feats[0], graphsage_aggregator_type)
                for etype, feature_dims in etype_source_dim.items()
            }, aggregate=heterogeneous_aggregation)
        ])

        for i in range(n_layers - 2):
            self._layers.append(
                HeteroGraphConv({
                    etype: dglnn.SAGEConv(hidden_feats[i], hidden_feats[i + 1], graphsage_aggregator_type)
                    for etype in etypes
                }, aggregate=heterogeneous_aggregation)
            )

        self._layers.append(
            HeteroGraphConv({
                etype: dglnn.SAGEConv(hidden_feats[-1], out_feats, graphsage_aggregator_type)
                for etype in etypes
            }, aggregate=heterogeneous_aggregation)
        )

        self._activation = activation

    @staticmethod
    def _validate_dimensionalities(hidden_feats: list[int], n_layers: Optional[int]) -> None:
        len_hidden_feats = len(hidden_feats)
        if n_layers is not None:
            if len_hidden_feats > n_layers - 1:
                raise ValueError(f"Too many hidden layers' dimensions specified, expected {n_layers - 1 = } to account for the output layer but got {len_hidden_feats}.")
            if len_hidden_feats < n_layers - 1:
                raise ValueError(f"Not enough hidden layers' dimensions specified, expected {n_layers - 1 = } to account for the output layer but got {len_hidden_feats}.")

    @staticmethod
    def _validate_all_nodes_have_known_dimensionalities(in_feats: dict[str, int], edge_definitions: set[tuple[str, str, str]]) -> None:
        for src_type, etype, dst_type in edge_definitions:
            assert src_type in in_feats, f"Source node of type '{src_type}' of edge '{etype}' does not have known feature dimensionality."
            assert dst_type in in_feats, f"Destination node of type '{dst_type}' of edge '{etype}' does not have known feature dimensionality."

    def forward(self, graph: DGLGraph, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h = inputs
        for layer in self._layers[:-1]:
            h = layer(graph, h)
            if self._activation:
                h = {k: self._activation(v) for k, v in h.items()}
        h = self._layers[-1](graph, h)
        return h
