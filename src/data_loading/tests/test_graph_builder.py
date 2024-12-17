import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
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

    def setup_method(self) -> None:
        self._dataset = TabularDataset(
            data_path=self._data_path,
            numeric_columns=[], categorical_columns=[], text_columns=[],
            required_columns=TEST_DATASET_ALL_COLUMNS,
        )

    def test_check_matching_definitions_fails_on_missing_nodes(self) -> None:
        with pytest.raises(AssertionError, match=r"Edge \('customer', 'makes', 'transaction'\) does not have its source column test_customer amongst node definition columns."):
            GraphDataset(source_tabular_dataset=self._dataset, node_feature_cols={'test_id': []}, node_label_cols={},
                         edge_definitions={('customer', 'makes', 'transaction'): ('test_customer', 'test_id')})

    def test_raises_error_on_multiple_columns_per_node_type(self) -> None:
        with pytest.raises(AssertionError, match='Node type customer is used as a reference to more than one column: test_customer, test_counterparty.'):
            GraphDataset(
                source_tabular_dataset=self._dataset,
                node_feature_cols={'test_id': [], 'test_customer': [], 'test_counterparty': []},
                node_label_cols={},
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
                node_label_cols={},
                edge_definitions={
                    ('customer', 'makes', 'tx'): ('test_customer', 'test_id'),
                    ('customer', 'makes', 'transaction'): ('test_customer', 'test_id'),
                },
            )

    def test_raises_warning_for_isolated_node_definitions(self, caplog: LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            GraphDataset(source_tabular_dataset=self._dataset, node_feature_cols={'test_id': [], 'test_customer': [], 'test_counterparty': []},
                         node_label_cols={}, edge_definitions={('customer', 'makes', 'transaction'): ('test_customer', 'test_id')})

        assert 'Nodes defined by column test_counterparty do not have any edges defined for them, will ignore these.' in caplog.text

    def test_basic_connectivity_build(self) -> None:
        graph_dataset = GraphDataset(
            source_tabular_dataset=self._dataset,
            node_feature_cols={'test_id': [], 'test_customer': [], 'test_counterparty': []},
            node_label_cols={},
            edge_definitions={
                ('customer', 'sends', 'transaction'): ('test_customer', 'test_id'),
                ('transaction', 'sent_to', 'counterparty'): ('test_id', 'test_counterparty'),
            },
        )
        graph_dataset.build_graph()

        test_graph = graph_dataset.graph

        assert test_graph is not None
        assert test_graph.num_nodes() == 15
        np.testing.assert_array_equal(test_graph.edges(etype='sends'), (torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1, 2, 3, 4])))
        np.testing.assert_array_equal(test_graph.edges(etype='sent_to'), (torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1, 2, 3, 4])))

    def test_basic_features_for_each_node_type(self) -> None:
        rnd_feature = np.random.rand(self._dataset.df.height)
        self._dataset.with_columns(
            pl.col('test_category').str.len_chars().alias('category_len'),
            pl.lit(rnd_feature).alias('random_feature'),
        )

        graph_dataset = GraphDataset(
            source_tabular_dataset=self._dataset,
            node_feature_cols={'test_id': ['test_amount'], 'test_customer': ['category_len'], 'test_counterparty': ['random_feature']},
            node_label_cols={},
            edge_definitions={
                ('customer', 'sends', 'transaction'): ('test_customer', 'test_id'),
                ('transaction', 'sent_to', 'counterparty'): ('test_id', 'test_counterparty'),
            },
        )
        graph_dataset.build_graph()

        assert graph_dataset.graph is not None
        assert list(graph_dataset.graph.ndata.keys()) ==  ['random_feature', 'category_len', 'test_amount']

        assert list(graph_dataset.graph.ndata['random_feature'].keys()) == ['counterparty']
        np.testing.assert_array_equal(graph_dataset.graph.ndata['random_feature']['counterparty'].flatten(), rnd_feature)

        assert list(graph_dataset.graph.ndata['category_len'].keys()) == ['customer']
        np.testing.assert_array_equal(graph_dataset.graph.ndata['category_len']['customer'].flatten(), torch.tensor([5, 5, 5, 5, 5]))

        assert list(graph_dataset.graph.ndata['test_amount'].keys()) == ['transaction']
        np.testing.assert_array_equal(graph_dataset.graph.ndata['test_amount']['transaction'].flatten(), torch.tensor([100, 200, 300, 400, 500]))

    def test_labels(self) -> None:
        labels = np.ones(self._dataset.df.height)
        self._dataset.with_columns(pl.lit(labels).alias('label'))

        graph_dataset = GraphDataset(
            source_tabular_dataset=self._dataset,
            node_feature_cols={'test_id': [], 'test_customer': []},
            node_label_cols={'test_id': 'label'},
            edge_definitions={('customer', 'sends', 'transaction'): ('test_customer', 'test_id')},
        )
        graph_dataset.build_graph()

        assert graph_dataset.graph is not None
        np.testing.assert_array_equal(graph_dataset.graph.ndata['label']['transaction'].flatten(), labels)

    def test_graph_update_connections(self) -> None:
        graph_dataset = GraphDataset(
            source_tabular_dataset=self._dataset,
            node_feature_cols={'test_id': [], 'test_customer': [], 'test_counterparty': []},
            node_label_cols={},
            edge_definitions={
                ('customer', 'sends', 'transaction'): ('test_customer', 'test_id'),
                ('transaction', 'sent_to', 'counterparty'): ('test_id', 'test_counterparty'),
            },
        )
        graph_dataset.build_graph()

        incr = pl.DataFrame({'test_id': [6, 7, 8], 'test_customer': ['B', 'C', 'F'], 'test_counterparty': ['ddd', 'eee', 'aaa']})
        graph_dataset.update_graph(incr)

        test_graph = graph_dataset.graph

        assert test_graph is not None
        assert test_graph.num_nodes() == 19
        np.testing.assert_array_equal(test_graph.edges(etype='sends'), (torch.tensor([0, 1, 2, 3, 4, 1, 2, 5]), torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])))
        np.testing.assert_array_equal(test_graph.edges(etype='sent_to'), (torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]), torch.tensor([0, 1, 2, 3, 4, 3, 4, 0])))

    def test_graph_update_new_nodes_with_features(self) -> None:
        rnd_feature = np.random.rand(self._dataset.df.height)
        self._dataset.with_columns(
            pl.col('test_category').str.len_chars().alias('category_len'),
            pl.lit(rnd_feature).alias('random_feature'),
        )

        graph_dataset = GraphDataset(
            source_tabular_dataset=self._dataset,
            node_feature_cols={'test_id': ['test_amount'], 'test_customer': ['category_len'], 'test_counterparty': ['random_feature']},
            node_label_cols={},
            edge_definitions={
                ('customer', 'sends', 'transaction'): ('test_customer', 'test_id'),
                ('transaction', 'sent_to', 'counterparty'): ('test_id', 'test_counterparty'),
            },
        )
        graph_dataset.build_graph()

        incr_rnd_feature = np.random.rand(3)
        incr = pl.DataFrame(
            {
                'test_id': [6, 7, 8], 'test_customer': ['F', 'G', 'H'], 'test_counterparty': ['fff', 'ggg', 'hhh'],
                'test_amount': [600, 700, 800], 'category_len': [6, 6, 6], 'random_feature': incr_rnd_feature,
            }
        )
        graph_dataset.update_graph(incr)

        test_graph = graph_dataset.graph
        assert test_graph is not None
        assert list(test_graph.ndata.keys()) ==  ['random_feature', 'category_len', 'test_amount']

        assert list(test_graph.ndata['random_feature'].keys()) == ['counterparty']
        np.testing.assert_array_equal(test_graph.ndata['random_feature']['counterparty'].flatten(), np.concatenate((rnd_feature, incr_rnd_feature)))

        assert list(test_graph.ndata['category_len'].keys()) == ['customer']
        np.testing.assert_array_equal(test_graph.ndata['category_len']['customer'].flatten(), torch.tensor([5, 5, 5, 5, 5, 6, 6, 6]))

        assert list(test_graph.ndata['test_amount'].keys()) == ['transaction']
        np.testing.assert_array_equal(test_graph.ndata['test_amount']['transaction'].flatten(), torch.tensor([100, 200, 300, 400, 500, 600, 700, 800]))

    def test_graph_update_new_nodes_with_labels(self) -> None:
        labels = np.ones(self._dataset.df.height)
        self._dataset.with_columns(pl.lit(labels).alias('label'))

        graph_dataset = GraphDataset(
            source_tabular_dataset=self._dataset,
            node_feature_cols={'test_id': [], 'test_customer': []},
            node_label_cols={'test_id': 'label'},
            edge_definitions={('customer', 'sends', 'transaction'): ('test_customer', 'test_id')},
        )
        graph_dataset.build_graph()

        incr = pl.DataFrame({'test_id': [6, 7, 8], 'test_customer': ['B', 'C', 'F'], 'label': [0, 0, 0]})
        graph_dataset.update_graph(incr)

        assert graph_dataset.graph is not None
        np.testing.assert_array_equal(graph_dataset.graph.ndata['label']['transaction'].flatten(), [1, 1, 1, 1, 1, 0, 0, 0])

    def test_graph_update_old_nodes_new_features(self) -> None:
        rnd_feature = np.random.rand(self._dataset.df.height)
        self._dataset.with_columns(
            pl.col('test_category').str.len_chars().alias('category_len'),
            pl.lit(rnd_feature).alias('random_feature'),
        )

        graph_dataset = GraphDataset(
            source_tabular_dataset=self._dataset,
            node_feature_cols={'test_id': ['test_amount'], 'test_customer': ['category_len'], 'test_counterparty': ['random_feature']},
            node_label_cols={},
            edge_definitions={
                ('customer', 'sends', 'transaction'): ('test_customer', 'test_id'),
                ('transaction', 'sent_to', 'counterparty'): ('test_id', 'test_counterparty'),
            },
        )
        graph_dataset.build_graph()

        incr = pl.DataFrame(
            {
                'test_id': [6], 'test_customer': ['B'], 'test_counterparty': ['aaa'],
                'test_amount': [600], 'category_len': [7], 'random_feature': [0.5],
            }
        )
        graph_dataset.update_graph(incr)
        rnd_feature[0] = 0.5

        test_graph = graph_dataset.graph
        assert test_graph is not None
        assert list(test_graph.ndata.keys()) ==  ['random_feature', 'category_len', 'test_amount']

        assert list(test_graph.ndata['random_feature'].keys()) == ['counterparty']
        np.testing.assert_array_equal(test_graph.ndata['random_feature']['counterparty'].flatten(), rnd_feature)

        assert list(test_graph.ndata['category_len'].keys()) == ['customer']
        np.testing.assert_array_equal(test_graph.ndata['category_len']['customer'].flatten(), torch.tensor([5, 7, 5, 5, 5]))

        assert list(test_graph.ndata['test_amount'].keys()) == ['transaction']
        np.testing.assert_array_equal(test_graph.ndata['test_amount']['transaction'].flatten(), torch.tensor([100, 200, 300, 400, 500, 600]))

    def test_graph_update_old_nodes_new_labels(self) -> None:
        labels = np.zeros(self._dataset.df.height)
        self._dataset.with_columns(pl.lit(labels).alias('label'))

        graph_dataset = GraphDataset(
            source_tabular_dataset=self._dataset,
            node_feature_cols={'test_id': [], 'test_customer': []},
            node_label_cols={'test_id': 'label'},
            edge_definitions={('customer', 'sends', 'transaction'): ('test_customer', 'test_id')},
        )
        graph_dataset.build_graph()

        incr = pl.DataFrame({'test_id': [1], 'test_customer': ['A'], 'label': [1]})
        graph_dataset.update_graph(incr)

        assert graph_dataset.graph is not None
        np.testing.assert_array_equal(graph_dataset.graph.ndata['label']['transaction'].flatten(), [1, 0, 0, 0, 0])

    def test_conversion_to_homogeneous(self) -> None:
        labels = np.ones(self._dataset.df.height)
        rnd_feature = np.random.rand(self._dataset.df.height)
        self._dataset.with_columns(
            pl.col('test_category').str.len_chars().alias('category_len'),
            pl.lit(rnd_feature).alias('random_feature'),
            pl.lit(labels).alias('label'),
        )

        graph_dataset = GraphDataset(
            source_tabular_dataset=self._dataset,
            node_feature_cols={'test_id': ['test_amount'], 'test_customer': ['category_len'], 'test_counterparty': ['random_feature']},
            node_label_cols={'test_id': 'label'},
            edge_definitions={
                ('customer', 'sends', 'transaction'): ('test_customer', 'test_id'),
                ('transaction', 'sent_to', 'counterparty'): ('test_id', 'test_counterparty'),
            },
        )
        graph_dataset.build_graph()

        test_homogeneous_graph = graph_dataset.get_homogeneous(store_type=False)
        expected_features = torch.zeros((15, 3))
        expected_features[0 : 5, 0] = torch.tensor(rnd_feature)
        expected_features[5 : 10, 1] = torch.tensor([5, 5, 5, 5, 5])
        expected_features[10 : 15, 2] = torch.tensor([100, 200, 300, 400, 500])
        expected_labels = torch.zeros(15)
        expected_labels[10 : 15] = torch.tensor([1, 1, 1, 1, 1])

        assert test_homogeneous_graph is not None
        assert test_homogeneous_graph.num_nodes() == 15
        np.testing.assert_array_equal(test_homogeneous_graph.ndata['_ID'], torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]))
        np.testing.assert_array_equal(test_homogeneous_graph.edata['_ID'], torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]))
        np.testing.assert_array_equal(test_homogeneous_graph.ndata['features'], expected_features)
        np.testing.assert_array_equal(test_homogeneous_graph.ndata['label'], expected_labels)
