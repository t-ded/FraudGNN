import logging
from typing import Optional, Any, cast, Iterator

import dgl
import polars as pl

from src.data_loading.tabular_dataset import TabularDataset

logger = logging.getLogger(__name__)


class GraphDataset:

    def __init__(self, source_tabular_dataset: TabularDataset, node_feature_cols: dict[str, list[str]], edge_definitions: dict[tuple[str, str, str], tuple[str, str]]) -> None:
        self._tabular_dataset = source_tabular_dataset
        self._node_defining_cols = list(node_feature_cols.keys())
        self._node_feature_cols = node_feature_cols
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

    def _assign_node_ids(self) -> None:
        self._tabular_dataset.with_columns(pl.int_range(0, pl.len()).alias('row_index'))
        for node_type, node_defining_col in self._node_type_to_column_name_mapping.items():
            self._value_node_id_mapping[node_type] = dict(cast(
                Iterator[tuple[Any, int]],
                self._tabular_dataset.ldf.select(node_defining_col, pl.col('row_index').first().over(node_defining_col)).collect().iter_rows()
            ))
            self._tabular_dataset.with_columns(pl.col(node_defining_col).replace_strict(self._value_node_id_mapping[node_type]))
