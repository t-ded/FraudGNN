import logging
from dataclasses import dataclass
from typing import Optional, Any, cast, Iterator

import dgl
import polars as pl
import torch

logger = logging.getLogger(__name__)


@dataclass
class GraphDatasetDefinition:
    node_feature_cols: dict[str, list[str]]
    label_node_col: Optional[str]
    label_col: Optional[str]
    edge_definitions: dict[tuple[str, str, str], tuple[str, str]]


class GraphDataset:

    def __init__(self, graph_dataset_definition: GraphDatasetDefinition) -> None:
        self._node_defining_cols = list(graph_dataset_definition.node_feature_cols.keys())
        self._node_feature_cols = graph_dataset_definition.node_feature_cols
        self._label_node_col = graph_dataset_definition.label_node_col
        self._label_col = graph_dataset_definition.label_col
        self._edge_definitions = graph_dataset_definition.edge_definitions

        self._node_type_to_column_name_mapping: dict[str, str] = {}
        self._column_name_to_node_type_mapping: dict[str, str] = {}

        self._validate()

        self._value_node_id_mapping: dict[str, dict[Any, int]] = {}
        self._graph: Optional[dgl.DGLGraph] = None
        self._features: dict[str, torch.Tensor] = {}
        self._labels: torch.Tensor = torch.empty(0, dtype=torch.float32)

    @property
    def graph(self) -> Optional[dgl.DGLGraph]:
        return self._graph

    @property
    def labels(self) -> torch.Tensor:
        return self._labels

    @property
    def label_node_col(self) -> Optional[str]:
        return self._label_node_col

    @property
    def label_col(self) -> Optional[str]:
        return self._label_col

    @property
    def node_features(self) -> dict[str, torch.Tensor]:
        return self._features

    def _validate(self) -> None:
        self._check_matching_node_edge_definitions()
        self._check_consistent_node_type_column_pairing()
        self._check_all_node_types_used_in_edgelist()
        self._check_label_col()

    def _check_matching_node_edge_definitions(self) -> None:
        for edge_description, edge_definition in self._edge_definitions.items():
            assert edge_definition[0] in self._node_defining_cols, f'Edge {edge_description} does not have its source column {edge_definition[0]} amongst node definition columns.'
            assert edge_definition[1] in self._node_defining_cols, f'Edge {edge_description} does not have its destination column {edge_definition[1]} amongst node definition columns.'

    def _check_consistent_node_type_column_pairing(self) -> None:

        for edge_description, edge_definition in self._edge_definitions.items():

            current_src_node_type, _, current_dst_node_type = edge_description
            current_src_column_name, current_dst_column_name = edge_definition

            known_column_name_for_src_node_type = self._node_type_to_column_name_mapping.get(current_src_node_type, None)
            assert known_column_name_for_src_node_type is None or known_column_name_for_src_node_type == current_src_column_name, \
                f'Node type {current_src_node_type} is used as a reference to more than one column: {known_column_name_for_src_node_type}, {current_src_column_name}.'

            known_column_name_for_dst_node_type = self._node_type_to_column_name_mapping.get(current_dst_node_type, None)
            assert known_column_name_for_dst_node_type is None or known_column_name_for_dst_node_type == current_dst_column_name, \
                f'Node type {current_dst_node_type} is used as a reference to more than one column: {known_column_name_for_dst_node_type}, {current_dst_column_name}.'

            known_node_type_for_src_column_name = self._column_name_to_node_type_mapping.get(current_src_column_name, None)
            assert known_node_type_for_src_column_name is None or known_node_type_for_src_column_name == current_src_node_type, \
                f'Column name {current_src_column_name} is referred to by more than one node type: {known_node_type_for_src_column_name}, {current_src_node_type}.'

            known_node_type_for_dst_column_name = self._column_name_to_node_type_mapping.get(current_dst_column_name, None)
            assert known_node_type_for_dst_column_name is None or known_node_type_for_dst_column_name == current_dst_node_type, \
                f'Column name {current_dst_column_name} is referred to by more than one node type: {known_node_type_for_dst_column_name}, {current_dst_node_type}.'

            self._node_type_to_column_name_mapping[current_src_node_type] = current_src_column_name
            self._node_type_to_column_name_mapping[current_dst_node_type] = current_dst_column_name
            self._column_name_to_node_type_mapping[current_src_column_name] = current_src_node_type
            self._column_name_to_node_type_mapping[current_dst_column_name] = current_dst_node_type

    def _check_all_node_types_used_in_edgelist(self) -> None:
        node_definitions_without_edge = set(self._node_defining_cols)
        for edge_definition in self._edge_definitions.values():
            node_definitions_without_edge.discard(edge_definition[0])
            node_definitions_without_edge.discard(edge_definition[1])

        for node_without_edges in node_definitions_without_edge:
            logger.warning(f'Nodes defined by column {node_without_edges} do not have any edges defined for them, will ignore these.')
            self._node_defining_cols.remove(node_without_edges)

    def _check_label_col(self) -> None:
        if (self._label_col is None) != (self._label_node_col is None):
            raise ValueError('Both label_col and label_node_col must be specified or neither should be.')

        if self._label_col is not None and self._label_node_col is not None:
            assert self._label_node_col in self._column_name_to_node_type_mapping, f'Node column {self._label_node_col} does not have a node type assigned to it via edgelist occurrence.'

    def get_ntype_for_column_name(self, column_name: str) -> str:
        return self._column_name_to_node_type_mapping[column_name]

    def build_graph(self, source_tabular_data: pl.LazyFrame) -> None:
        source_tabular_data = self._assign_node_ids(source_tabular_data)

        self._graph = dgl.heterograph(
            data_dict={
                edge_description: tuple(source_tabular_data.select(edge_definition).collect().to_numpy(writable=True).T)
                for edge_description, edge_definition in self._edge_definitions.items()
            },
            num_nodes_dict={
                node_type: (
                    source_tabular_data.select(pl.col(node_defining_col).n_unique()).collect().item()
                    if node_defining_col != self._label_node_col
                    else source_tabular_data.select(pl.len()).collect().item()
                )
                for node_type, node_defining_col in self._node_type_to_column_name_mapping.items()
            }
        )

        self._enrich_with_features(source_tabular_data)
        if self._label_col is not None and self._label_node_col is not None:
            self._enrich_with_labels(source_tabular_data)

    def _assign_node_ids(self, source_tabular_data: pl.LazyFrame) -> pl.LazyFrame:
        for node_type, node_defining_col in self._node_type_to_column_name_mapping.items():
            self._value_node_id_mapping[node_type] = dict(cast(
                Iterator[tuple[Any, int]],
                source_tabular_data.select(node_defining_col, pl.col(node_defining_col).rank('dense').sub(1).alias('node_id')).collect().iter_rows()
            ))
            source_tabular_data = source_tabular_data.with_columns(pl.col(node_defining_col).replace_strict(self._value_node_id_mapping[node_type]))
        return source_tabular_data

    def _enrich_with_features(self, source_tabular_data: pl.LazyFrame) -> None:
        assert isinstance(self._graph, dgl.DGLGraph), 'Can only enrich with features after graph has been initialized.'
        for node_col, node_feature_cols in self._node_feature_cols.items():
            node_type = self._column_name_to_node_type_mapping[node_col]

            pre_selection = source_tabular_data
            if node_col != self._label_node_col:
                pre_selection = pre_selection.unique(node_col, maintain_order=True)
            node_feature_df = pre_selection.select(node_feature_cols).collect()
            if node_feature_df.is_empty():
                node_features = torch.empty(node_feature_df.shape, dtype=torch.float32)
            else:
                node_features = node_feature_df.to_torch().type(torch.float32)
            self._features[node_type] = node_features

    def _enrich_with_labels(self, source_tabular_data: pl.LazyFrame) -> None:
        assert isinstance(self._graph, dgl.DGLGraph), 'Can only enrich with labels after graph has been initialized.'
        self._labels = source_tabular_data.select(self._label_col).collect().to_torch().type(torch.float32)

    def update_graph(self, incr: pl.DataFrame) -> None:
        assert isinstance(self._graph, dgl.DGLGraph), 'Can only update graph after graph has been initialized.'

        for node_type, node_defining_col in self._node_type_to_column_name_mapping.items():
            incr = self._assign_node_ids_incr(incr.lazy(), node_type, node_defining_col).collect()
            new_id_mapping = dict(cast(
                Iterator[tuple[Any, int]],
                incr.filter((pl.col('is_new_node_mask'))).select('cached_col', node_defining_col).iter_rows()
            ))

            self._value_node_id_mapping[node_type].update(new_id_mapping)

            self._update_ntype_features_from_incr(node_type, node_defining_col, incr)
            if new_id_mapping:
                self._graph.add_nodes(
                    num=len(new_id_mapping),
                    data=None,
                    ntype=node_type,
                )
            incr = incr.drop('cached_col', 'is_new_node_mask')

        self._update_edges_from_incr(incr)
        if self._label_col is not None and self._label_node_col is not None:
            self._update_labels_from_incr(incr)

    def _assign_node_ids_incr(self, incr_lazy: pl.LazyFrame, node_type: str, node_defining_col: str) -> pl.LazyFrame:
        max_id = max(self._value_node_id_mapping[node_type].values(), default=-1)
        return (
            incr_lazy
            .with_columns(
                pl.col(node_defining_col).alias('cached_col'),
                pl.col(node_defining_col).replace_strict(self._value_node_id_mapping[node_type], default=None)
            )
            .with_columns(
                pl.col(node_defining_col).is_null().alias('is_new_node_mask'),
            )
            .with_columns(
                pl.col(node_defining_col).fill_null(pl.col('cached_col').rank('dense').over('is_new_node_mask').add(max_id))
            )
        )

    def _update_ntype_features_from_incr(self, node_type: str, node_defining_col: str, incr: pl.DataFrame) -> None:
        assert isinstance(self._graph, dgl.DGLGraph), 'Can only update features after graph has been initialized.'

        if node_defining_col != self._label_node_col:
            incr = incr.unique(node_defining_col, maintain_order=True)

        incr_new = incr.filter(pl.col('is_new_node_mask'))
        incr_old = incr.filter(~pl.col('is_new_node_mask'))
        old_nodes_ids = incr_old.select(node_defining_col).to_torch().flatten()

        num_new = len(incr_new)
        num_features_for_this_node_type = len(self._node_feature_cols[node_defining_col])
        new_nodes_data: torch.Tensor = torch.empty((num_new, num_features_for_this_node_type), dtype=torch.float32)

        if len(old_nodes_ids) > 0:
            for nth_feature, feature_col in enumerate(self._node_feature_cols[node_defining_col]):
                self._features[node_type][old_nodes_ids, nth_feature] = incr_old.select(feature_col).to_torch().type(torch.float32).flatten()
                new_nodes_data[:, nth_feature] = incr_new.select(feature_col).to_torch().type(torch.float32).flatten()
        else:
            for nth_feature, feature_col in enumerate(self._node_feature_cols[node_defining_col]):
                new_nodes_data[:, nth_feature] = incr_new.select(feature_col).to_torch().type(torch.float32).flatten()

        self._update_ntype_with_features(node_type, new_nodes_data)

    def _update_ntype_with_features(self, ntype: str, feature_values: torch.Tensor) -> None:
        self._features[ntype] = torch.cat((self._features[ntype], feature_values), dim=0)

    def _update_edges_from_incr(self, incr: pl.DataFrame) -> None:
        assert isinstance(self._graph, dgl.DGLGraph), 'Can only update graph after graph has been initialized.'

        for edge_description, edge_definition in self._edge_definitions.items():
            edge_src_dst = incr.select(edge_definition).to_numpy(writable=True).T
            self._graph.add_edges(*edge_src_dst, etype=edge_description[1])

    def _update_labels_from_incr(self, incr: pl.DataFrame) -> None:
        self._labels = torch.cat([self._labels, incr.select(self._label_col).to_torch().type(torch.float32)], dim=0)

    def get_homogeneous(self, store_type: bool = True) -> dgl.DGLGraph:
        assert isinstance(self._graph, dgl.DGLGraph), 'Can only convert graph to homogeneous after graph has been initialized.'

        homo_g = dgl.to_homogeneous(self._graph, store_type=store_type)

        num_all_feature_cols = sum(len(feature_cols) for feature_cols in self._node_feature_cols.values())
        num_nodes = homo_g.num_nodes()
        homogeneous_features = torch.zeros((num_nodes, num_all_feature_cols))
        homogeneous_labels = torch.zeros(num_nodes)

        node_idx = 0
        feature_idx = 0
        label_node_idx = 0
        for ntype in self._graph.ntypes:

            ntype_col = self._node_type_to_column_name_mapping[ntype]
            ntype_num_nodes = self._graph.num_nodes(ntype)
            ntype_features = self._features[ntype]
            homogeneous_features[node_idx : node_idx + ntype_num_nodes, feature_idx : feature_idx + ntype_features.shape[1]] = ntype_features
            feature_idx += ntype_features.shape[1]

            if ntype_col == self._label_node_col:
                homogeneous_labels[label_node_idx : label_node_idx + ntype_num_nodes] = self._labels.squeeze()

            node_idx += ntype_num_nodes
            label_node_idx += ntype_num_nodes

        homo_g.ndata['features'] = homogeneous_features
        homo_g.ndata['label'] = homogeneous_labels
        return homo_g
