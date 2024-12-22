import json
from pathlib import Path

import pytest
from torch import nn

from src.base.build_test_dataset import ENRICHED_TEST_DATASET_ALL_COLUMNS
from src.data_loading.dynamic_dataset import DynamicDataset
from src.data_loading.graph_builder import GraphDatasetDefinition
from src.data_loading.tabular_dataset import TabularDatasetDefinition, TrainValTestRatios
from src.models.heterogeneous_graphsage import HeteroGraphSAGE

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = ROOT_DIR / 'config/general_config.json'

with open(CONFIG_PATH) as f:
    config = json.load(f)

MODELS_TO_TEST = [HeteroGraphSAGE]

@pytest.fixture(params=MODELS_TO_TEST)
def model(request) -> nn.Module:
    return request.param


class TestModels:

    _data_path = ROOT_DIR / config['test_data_path'] / 'enriched_minimal_sample.csv'

    def setup_method(self) -> None:

        tabular_definition = TabularDatasetDefinition(
            data_path=self._data_path,
            numeric_columns=['test_amount'], categorical_columns=['category_len'], text_columns=[],
            required_columns=ENRICHED_TEST_DATASET_ALL_COLUMNS,
            train_val_test_ratios=TrainValTestRatios(0.5, 0.25, 0.25),
        )
        graph_definition = GraphDatasetDefinition(
            node_feature_cols={'test_id': ['test_amount', 'category_len', 'random_feature'], 'test_customer': ['category_len'], 'test_counterparty': ['random_feature', 'category_len']},
            node_label_cols={'test_id': 'label'},
            edge_definitions={
                ('customer', 'sends', 'transaction'): ('test_customer', 'test_id'),
                ('transaction', 'sent_to', 'counterparty'): ('test_id', 'test_counterparty'),
                ('counterparty', 'same_id_as_1', 'customer'): ('test_counterparty', 'test_customer'),
                ('customer', 'same_id_as_2', 'counterparty'): ('test_customer', 'test_counterparty'),
            },
            unique_cols={'test_id'},
        )
        self._dynamic_dataset = DynamicDataset(name='TestDynamicDataset', tabular_dataset_definition=tabular_definition, graph_dataset_definition=graph_definition)
        self._in_feats = {ntype: features.shape[1] for ntype, features in self._dynamic_dataset.graph_features.items()}

    def test_raises_error_when_hidden_feats_constant_n_layers_none(self, model: nn.Module) -> None:
        with pytest.raises(AssertionError, match="Hidden layers' dimensions were given as a constant but number of layers was not specified."):
            model(in_feats={}, hidden_feats=5, n_layers=None, out_feats=1, edge_definitions=set())

    def test_raises_error_too_many_hidden_feats(self, model: nn.Module) -> None:
        with pytest.raises(ValueError, match="Too many hidden layers' dimensions specified, expected n_layers - 1 = 3 to account for the output layer but got 5."):
            model(in_feats={}, hidden_feats=[16, 16, 16, 16, 16], n_layers=4, out_feats=1, edge_definitions=set())

    def test_raises_error_not_enough_hidden_feats(self, model: nn.Module) -> None:
        with pytest.raises(ValueError, match="Not enough hidden layers' dimensions specified, expected n_layers - 1 = 3 to account for the output layer but got 2."):
            model(in_feats={}, hidden_feats=[16, 16], n_layers=4, out_feats=1, edge_definitions=set())

    def test_raises_error_inconsistent_edge_definitions_in_feats(self, model: nn.Module) -> None:
        with pytest.raises(AssertionError, match="Source node of type 'user' of edge 'follows' does not have known feature dimensionality."):
            model(in_feats={'content': 4}, hidden_feats=5, n_layers=1, out_feats=1, edge_definitions={('user', 'follows', 'content')})
        with pytest.raises(AssertionError, match="Destination node of type 'content' of edge 'follows' does not have known feature dimensionality."):
            model(in_feats={'user': 4}, hidden_feats=5, n_layers=1, out_feats=1, edge_definitions={('user', 'follows', 'content')})

    def test_forward_pass(self, model: nn.Module) -> None:
        assert self._dynamic_dataset.graph is not None

        model_instance = model(in_feats=self._in_feats, hidden_feats=[16, 16], n_layers=3, out_feats=1, edge_definitions=set(self._dynamic_dataset.graph.canonical_etypes))
        res = model_instance.forward(self._dynamic_dataset.graph, self._dynamic_dataset.graph_features)
        for ntype in self._dynamic_dataset.graph.ntypes:
            assert ntype in res
            assert list(res[ntype].shape) == [3, 1]
