import json
from pathlib import Path

import dgl.nn as dglnn
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from torch import nn

from src.base.build_test_dataset import ENRICHED_TEST_DATASET_ALL_COLUMNS
from src.data_loading.graph_builder import GraphDatasetDefinition
from src.data_loading.tabular_dataset import TabularDatasetDefinition, TrainValTestRatios
from src.evaluation.evaluator import Evaluator, GNNHyperparameters

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = ROOT_DIR / 'config/general_config.json'

with open(CONFIG_PATH) as f:
    config = json.load(f)


class NaiveRGCN(nn.Module):
    def __init__(self, in_feats: int, hid_feats: int, out_feats: int, rel_names: list[str]) -> None:
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph: DGLGraph, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class TestEvaluator:

    _data_path = ROOT_DIR / config['test_data_path'] / 'enriched_minimal_sample.csv'

    def setup_method(self) -> None:

        tabular_definition = TabularDatasetDefinition(
            data_path=self._data_path,
            numeric_columns=['test_amount'], categorical_columns=['category_len'], text_columns=[],
            required_columns=ENRICHED_TEST_DATASET_ALL_COLUMNS,
            train_val_test_ratios=TrainValTestRatios(0.5, 0.25, 0.25),
        )
        graph_definition = GraphDatasetDefinition(
            node_feature_cols={'test_id': ['test_amount'], 'test_customer': ['category_len'], 'test_counterparty': ['random_feature']},
            node_label_cols={'test_id': 'label'},
            edge_definitions={
                ('customer', 'sends', 'transaction'): ('test_customer', 'test_id'),
                ('transaction', 'sent_to', 'counterparty'): ('test_id', 'test_counterparty'),
                ('counterparty', 'same_id_as', 'customer'): ('test_counterparty', 'test_customer'),
            },
            unique_cols={'test_id'},
        )
        self._model = NaiveRGCN(1, 2, 1, [key[1] for key in graph_definition.edge_definitions.keys()])

        self._evaluator = Evaluator(
            model=self._model,
            hyperparameters=GNNHyperparameters(
                learning_rate=0.01,
                batch_size=1,
            ),
            tabular_dataset_definition=tabular_definition,
            graph_dataset_definition=graph_definition,
            identifier='GNNTest'
        )

    def test_train(self) -> None:
        self._evaluator.train(num_epochs=1)
        assert True

    def test_validate(self) -> None:
        evaluation_results = self._evaluator.stream_evaluate('validation')
        assert evaluation_results.total_loss(nn.BCEWithLogitsLoss()) >= 0
        assert evaluation_results.precision == 0.0
        assert evaluation_results.recall == 0.0
        assert evaluation_results.num_samples == 1

    def test_testing(self) -> None:
        evaluation_results = self._evaluator.stream_evaluate('testing')
        assert evaluation_results.num_samples == 1
        assert evaluation_results.precision >= 0.0
        assert evaluation_results.recall >= 0.0
