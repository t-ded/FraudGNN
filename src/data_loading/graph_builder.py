import logging
from dataclasses import dataclass
from typing import Optional, Any, cast, Iterator

import dgl
import polars as pl
import torch
from dgl import DGLGraph

from src.data_loading.tabular_dataset import TabularDataset

logger = logging.getLogger(__name__)


@dataclass
class GraphDatasetDefinition:
    node_feature_cols: dict[str, list[str]]
    node_label_cols: dict[str, str]
    edge_definitions: dict[tuple[str, str, str], tuple[str, str]]
    batch_size: int


class GraphDataset:

    def __init__(self, graph_dataset_definition: GraphDatasetDefinition) -> None:
        self._node_defining_cols = list(graph_dataset_definition.node_feature_cols.keys())
        self._node_feature_cols = graph_dataset_definition.node_feature_cols
        self._node_label_cols = graph_dataset_definition.node_label_cols
        self._edge_definitions = graph_dataset_definition.edge_definitions

        self._node_type_to_column_name_mapping: dict[str, str] = {}
        self._column_name_to_node_type_mapping: dict[str, str] = {}

        self._validate()

        self._value_node_id_mapping: dict[str, dict[Any, int]] = {}
        self._graph: Optional[dgl.DGLGraph] = None

    @property
    def graph(self) -> Optional[dgl.DGLGraph]:
        return self._graph

    def _validate(self) -> None:
        self._check_matching_node_edge_definitions()
        self._check_consistent_node_type_column_pairing()
        self._check_all_node_types_used_in_edgelist()

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

    def build_graph(self, source_tabular_dataset: TabularDataset) -> None:
        self._assign_node_ids(source_tabular_dataset)

        self._graph = dgl.heterograph(
            data_dict={
                edge_description: tuple(source_tabular_dataset.ldf.select(edge_definition).collect().to_numpy(writable=True).T)
                for edge_description, edge_definition in self._edge_definitions.items()
            },
            num_nodes_dict={
                node_type: source_tabular_dataset.ldf.select(pl.col(node_defining_col).n_unique()).collect().item()
                for node_type, node_defining_col in self._node_type_to_column_name_mapping.items()
            }
        )

        self._enrich_with_features(source_tabular_dataset)
        self._enrich_with_labels(source_tabular_dataset)

    def _assign_node_ids(self, source_tabular_dataset: TabularDataset) -> None:
        source_tabular_dataset.with_columns(pl.int_range(0, pl.len()).alias('row_index'))
        for node_type, node_defining_col in self._node_type_to_column_name_mapping.items():
            self._value_node_id_mapping[node_type] = dict(cast(
                Iterator[tuple[Any, int]],
                source_tabular_dataset.ldf.select(node_defining_col, pl.col('row_index').first().over(node_defining_col)).collect().iter_rows()
            ))
            source_tabular_dataset.with_columns(pl.col(node_defining_col).replace_strict(self._value_node_id_mapping[node_type]))

    def _enrich_with_features(self, source_tabular_dataset: TabularDataset) -> None:
        assert isinstance(self._graph, DGLGraph), 'Can only enrich with features after graph has been initialized.'
        for node_col, node_feature_cols in self._node_feature_cols.items():
            node_type = self._column_name_to_node_type_mapping[node_col]
            for feature_col in node_feature_cols:
                self._graph.nodes[node_type].data[feature_col] = source_tabular_dataset.ldf.select(feature_col).collect().to_torch()

    def _enrich_with_labels(self, source_tabular_dataset: TabularDataset) -> None:
        assert isinstance(self._graph, DGLGraph), 'Can only enrich with labels after graph has been initialized.'
        for node_col, label_col in self._node_label_cols.items():
            node_type = self._column_name_to_node_type_mapping[node_col]
            self._graph.nodes[node_type].data[label_col] = source_tabular_dataset.ldf.select(label_col).collect().to_torch().long()

    def update_graph(self, incr: pl.DataFrame) -> None:
        assert isinstance(self._graph, DGLGraph), 'Can only update graph after graph has been initialized.'

        for node_type, node_defining_col in self._node_type_to_column_name_mapping.items():
            incr = self._assign_node_ids_incr(incr.lazy(), node_type, node_defining_col).collect()
            new_id_mapping = dict(cast(
                Iterator[tuple[Any, int]],
                incr.filter((pl.col('is_new_node_mask'))).select('cached_col', node_defining_col).iter_rows()
            ))

            self._value_node_id_mapping[node_type].update(new_id_mapping)

            new_nodes_data = self._get_new_data_update_old_data(node_type, node_defining_col, incr)
            if new_id_mapping:
                self._graph.add_nodes(
                    num=len(new_id_mapping),
                    data=new_nodes_data if new_nodes_data else None,
                    ntype=node_type,
                )

        self._update_edges_from_incr(incr)

    def _assign_node_ids_incr(self, incr_lazy: pl.LazyFrame, node_type: str, node_defining_col: str) -> pl.LazyFrame:
        max_id = max(self._value_node_id_mapping[node_type].values(), default=-1) + 1
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
                pl.int_range(max_id, pl.len() + max_id).over('is_new_node_mask').alias('new_idx')
            )
            .with_columns(
                pl.col(node_defining_col).fill_null(pl.col('new_idx').first().over('cached_col'))
            )
        )

    def _get_new_data_update_old_data(self, node_type: str, node_defining_col: str, incr: pl.DataFrame) -> dict[str, torch.Tensor]:
        assert isinstance(self._graph, DGLGraph), 'Can only update graph after graph has been initialized.'

        incr_new = incr.filter(pl.col('is_new_node_mask'))
        incr_old = incr.filter(~pl.col('is_new_node_mask'))

        old_nodes_ids = incr_old.select(node_defining_col).to_torch()
        new_nodes_data: dict[str, torch.Tensor] = {}
        label_col = self._node_label_cols.get(node_defining_col)

        if len(old_nodes_ids) > 0:

            for feature_col in self._node_feature_cols[node_defining_col]:
                self._graph.nodes[node_type].data[feature_col][old_nodes_ids] = incr_old.select(feature_col).to_torch()
                new_nodes_data[feature_col] = incr_new.select(feature_col).to_torch()

            if label_col is not None:
                self._graph.nodes[node_type].data[label_col][old_nodes_ids] = incr_old.select(label_col).to_torch()
                new_nodes_data[label_col] = incr_new.select(label_col).to_torch()

        else:

            for feature_col in self._node_feature_cols[node_defining_col]:
                new_nodes_data[feature_col] = incr_new.select(feature_col).to_torch()

            if label_col is not None:
                new_nodes_data[label_col] = incr_new.select(label_col).to_torch()

        return new_nodes_data

    def _update_edges_from_incr(self, incr: pl.DataFrame) -> None:
        assert isinstance(self._graph, DGLGraph), 'Can only update graph after graph has been initialized.'

        for edge_description, edge_definition in self._edge_definitions.items():
            edge_src_dst = incr.select(edge_definition).to_numpy(writable=True).T
            self._graph.add_edges(*edge_src_dst, etype=edge_description[1])

    def get_homogeneous(self, store_type: bool = True) -> dgl.DGLGraph:
        assert isinstance(self._graph, DGLGraph), 'Can only convert graph to homogeneous after graph has been initialized.'

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

            for feature_col in self._node_feature_cols[ntype_col]:
                homogeneous_features[node_idx : node_idx + ntype_num_nodes, feature_idx] = self._graph.nodes[ntype].data[feature_col].squeeze(1)
                feature_idx += 1

            label_col = self._node_label_cols.get(ntype_col)
            if label_col is not None:
                homogeneous_labels[label_node_idx : label_node_idx + ntype_num_nodes] = self._graph.nodes[ntype].data[label_col].squeeze(1)

            node_idx += ntype_num_nodes
            label_node_idx += ntype_num_nodes

        homo_g.ndata['features'] = homogeneous_features
        homo_g.ndata['label'] = homogeneous_labels
        return homo_g
