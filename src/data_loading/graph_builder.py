import logging
from typing import Optional, Any, cast, Iterator

import dgl
import polars as pl
import torch
from dgl import DGLGraph

from src.data_loading.tabular_dataset import TabularDataset

logger = logging.getLogger(__name__)


class GraphDataset:

    def __init__(
            self,
            source_tabular_dataset: TabularDataset,
            node_feature_cols: dict[str, list[str]],
            node_label_cols: dict[str, str],
            edge_definitions: dict[tuple[str, str, str], tuple[str, str]],
    ) -> None:
        self._tabular_dataset = source_tabular_dataset
        self._node_defining_cols = list(node_feature_cols.keys())
        self._node_feature_cols = node_feature_cols
        self._node_label_cols = node_label_cols
        self._edge_definitions = edge_definitions

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

    def build_graph(self) -> None:
        self._assign_node_ids()

        self._graph = dgl.heterograph(
            data_dict={
                edge_description: tuple(self._tabular_dataset.ldf.select(edge_definition).collect().to_numpy(writable=True).T)
                for edge_description, edge_definition in self._edge_definitions.items()
            },
            num_nodes_dict={
                node_type: self._tabular_dataset.ldf.select(pl.col(node_defining_col).n_unique()).collect().item()
                for node_type, node_defining_col in self._node_type_to_column_name_mapping.items()
            }
        )

        self._enrich_with_features()
        self._enrich_with_labels()

    def _assign_node_ids(self) -> None:
        self._tabular_dataset.with_columns(pl.int_range(0, pl.len()).alias('row_index'))
        for node_type, node_defining_col in self._node_type_to_column_name_mapping.items():
            self._value_node_id_mapping[node_type] = dict(cast(
                Iterator[tuple[Any, int]],
                self._tabular_dataset.ldf.select(node_defining_col, pl.col('row_index').first().over(node_defining_col)).collect().iter_rows()
            ))
            self._tabular_dataset.with_columns(pl.col(node_defining_col).replace_strict(self._value_node_id_mapping[node_type]))

    def _enrich_with_features(self) -> None:
        assert isinstance(self._graph, DGLGraph), 'Can only enrich with features after graph has been initialized.'
        for node_col, node_feature_cols in self._node_feature_cols.items():
            node_type = self._column_name_to_node_type_mapping[node_col]
            for feature_col in node_feature_cols:
                self._graph.nodes[node_type].data[feature_col] = self._tabular_dataset.ldf.select(feature_col).collect().to_torch()

    def _enrich_with_labels(self) -> None:
        assert isinstance(self._graph, DGLGraph), 'Can only enrich with labels after graph has been initialized.'
        for node_col, label_col in self._node_label_cols.items():
            node_type = self._column_name_to_node_type_mapping[node_col]
            self._graph.nodes[node_type].data[label_col] = self._tabular_dataset.ldf.select(label_col).collect().to_torch().long()

    def update_graph(self, incr: pl.DataFrame) -> None:
        assert isinstance(self._graph, DGLGraph), 'Can only update graph after graph has been initialized.'

        for node_type, node_defining_col in self._node_type_to_column_name_mapping.items():

            existing_ids = self._value_node_id_mapping[node_type]
            max_id = max(existing_ids.values(), default=-1) + 1

            new_values = incr.select(pl.col(node_defining_col)).unique(keep='first', maintain_order=True).to_series()
            new_id_mapping = {}
            for value in new_values:
                if value not in existing_ids:
                    new_id_mapping[value] = max_id
                    max_id += 1
            self._value_node_id_mapping[node_type].update(new_id_mapping)
            incr = incr.with_columns(pl.col(node_defining_col).replace_strict(self._value_node_id_mapping[node_type]))

            node_type_update_data: dict[str, torch.Tensor] = {}

            for feature_col in self._node_feature_cols[node_defining_col]:
                node_type_update_data[feature_col] = incr.select(feature_col).to_torch()

            label_col = self._node_label_cols.get(node_defining_col)
            if label_col is not None:
                node_type_update_data[label_col] = incr.select(label_col).to_torch()

            if new_id_mapping:
                self._graph.add_nodes(
                    num=len(new_id_mapping),
                    data=node_type_update_data if node_type_update_data else None,
                    ntype=node_type,
                )

        for edge_description, edge_definition in self._edge_definitions.items():
            self._graph.add_edges(
                *tuple(incr.select(edge_definition).to_numpy(writable=True).T),
                etype=edge_description[1],
            )

    def get_homogeneous(self, store_type: bool = True) -> dgl.DGLGraph:
        return dgl.to_homogeneous(self._graph, store_type=store_type)
