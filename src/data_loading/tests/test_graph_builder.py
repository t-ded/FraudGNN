import json
import logging
from pathlib import Path

import numpy as np
import pytest
import torch
from _pytest.logging import LogCaptureFixture

from src.base.build_test_dataset import TEST_DATASET_ALL_COLUMNS
from src.data_loading.graph_builder import GraphDataset
from src.data_loading.tabular_dataset import TabularDataset

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = ROOT_DIR / 'config/general_config.json'

with open(CONFIG_PATH) as f:
    config = json.load(f)


class TestGraphDataset:

    _data_path = ROOT_DIR / config['test_data_path'] / 'minimal_sample.csv'
    _dataset = TabularDataset(
        data_path=_data_path,
        numeric_columns=[], categorical_columns=[], text_columns=[],
        required_columns=TEST_DATASET_ALL_COLUMNS,
    )

    def test_check_matching_definitions_fails_on_missing_nodes(self) -> None:
        with pytest.raises(AssertionError, match=r"Edge \('customer', 'makes', 'transaction'\) does not have its source column test_customer amongst node definition columns."):
            GraphDataset(source_tabular_dataset=self._dataset, node_feature_cols={'test_id': []}, edge_definitions={('customer', 'makes', 'transaction'): ('test_customer', 'test_id')})

    def test_raises_error_on_multiple_columns_per_node_type(self) -> None:
        with pytest.raises(AssertionError, match='Node type customer is used as a reference to more than one column: test_customer, test_counterparty.'):
            GraphDataset(
                source_tabular_dataset=self._dataset,
                node_feature_cols={'test_id': [], 'test_customer': [], 'test_counterparty': []},
                edge_definitions={
                    ('customer', 'makes', 'transaction'): ('test_customer', 'test_id'),
                    ('customer', 'does', 'transaction'): ('test_counterparty', 'test_id'),
                },
            )

    def test_raises_error_on_multiple_node_types_per_column(self) -> None:
        with pytest.raises(AssertionError, match='Column name test_id is referred to by more than one node type: tx, transaction.'):
            GraphDataset(
                source_tabular_dataset=self._dataset,
                node_feature_cols={'test_id': [], 'test_customer': [], 'test_counterparty': []},
                edge_definitions={
                    ('customer', 'makes', 'tx'): ('test_customer', 'test_id'),
                    ('customer', 'makes', 'transaction'): ('test_customer', 'test_id'),
                },
            )

    def test_raises_warning_for_isolated_node_definitions(self, caplog: LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            GraphDataset(source_tabular_dataset=self._dataset, node_feature_cols={'test_id': [], 'test_customer': [], 'test_counterparty': []},
                         edge_definitions={('customer', 'makes', 'transaction'): ('test_customer', 'test_id')})

        assert 'Nodes defined by column test_counterparty do not have any edges defined for them, will ignore these.' in caplog.text

    def test_basic_connectivity_build(self) -> None:
        graph_dataset = GraphDataset(
            source_tabular_dataset=self._dataset,
            node_feature_cols={'test_id': [], 'test_customer': [], 'test_counterparty': []},
            edge_definitions={
                ('customer', 'sends', 'transaction'): ('test_customer', 'test_id'),
                ('transaction', 'sent_to', 'counterparty'): ('test_id', 'test_counterparty'),
            },
        )
        graph_dataset.build_graph()

        test_graph = graph_dataset.graph

        assert test_graph is not None
        np.testing.assert_array_equal(test_graph.edges(etype='sends'), (torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1, 2, 3, 4])))
        np.testing.assert_array_equal(test_graph.edges(etype='sent_to'), (torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1, 2, 3, 4])))
