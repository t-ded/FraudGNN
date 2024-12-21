import json
from pathlib import Path

import dgl.nn as dglnn
import pytest
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from torch import nn

from src.base.build_test_dataset import ENRICHED_TEST_DATASET_ALL_COLUMNS
from src.data_loading.dynamic_dataset import DynamicDataset
from src.data_loading.graph_builder import GraphDatasetDefinition
from src.data_loading.tabular_dataset import TabularDatasetDefinition, TrainValTestRatios
from src.evaluation.evaluator import Evaluator, GNNHyperparameters

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = ROOT_DIR / 'config/general_config.json'

with open(CONFIG_PATH) as f:
    config = json.load(f)


class RGCN(nn.Module):
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

        self._dataset = DynamicDataset(
            name='test_dataset',
            tabular_dataset_definition=TabularDatasetDefinition(
                data_path=self._data_path,
                numeric_columns=['test_amount'], categorical_columns=['category_len'], text_columns=[],
                required_columns=ENRICHED_TEST_DATASET_ALL_COLUMNS,
                train_val_test_ratios=TrainValTestRatios(0.5, 0.25, 0.25),
            ),
            graph_dataset_definition=GraphDatasetDefinition(
                node_feature_cols={'test_id': ['test_amount'], 'test_customer': ['category_len'], 'test_counterparty': ['random_feature']},
                node_label_cols={'test_id': 'label'},
                edge_definitions={
                    ('customer', 'sends', 'transaction'): ('test_customer', 'test_id'),
                    ('transaction', 'sent_to', 'counterparty'): ('test_id', 'test_counterparty'),
                    ('counterparty', 'same_id_as', 'customer'): ('test_counterparty', 'test_customer'),
                },
            ),
        )

        assert self._dataset.graph is not None
        self._model = RGCN(1, 2, 1, self._dataset.graph.etypes)

        self._evaluator = Evaluator(
            model=self._model,
            dynamic_dataset=self._dataset,
            hyperparameters=GNNHyperparameters(
                learning_rate=0.01,
                batch_size=1,
            ),
        )

    def test_train(self):
        self._evaluator.train(num_epochs=1)
        assert True
    #
    # def test_validate(self):
    #     """
    #     Tests that the validation function calculates a loss value.
    #     """
    #     loss = self._evaluator.validate()
    #     assert loss >= 0, "Validation loss should be non-negative."
    #
    # def test_stream_evaluate(self):
    #     """
    #     Tests that streaming evaluation processes batches correctly.
    #     """
    #     data_loader = DataLoader(self.dummy_data, batch_size=1)
    #     results = self._evaluator.stream_evaluate(data_loader)
    #     assert isinstance(results, list), "Streaming evaluation should return a list."
    #     assert 'overall_loss' in results[-1], "Streaming evaluation results should include overall loss."
    #
    # def test_get_labels(self):
    #     """
    #     Tests the _get_labels function to ensure it extracts labels correctly.
    #     """
    #     labels = self._evaluator._get_labels(self.graph)
    #     assert torch.equal(labels, self.graph.nodes['item'].data['label']), "Labels should match the ground truth."
    #
    # def test_device_setting(self):
    #     """
    #     Tests if the model and data are moved to the appropriate device.
    #     """
    #     device = self._evaluator.device
    #     assert self._model.conv1['likes'].weight.device.type == device, f"Model weights should be on device {device}."
    #
    # def test_training_with_no_labels(self):
    #     """
    #     Ensures that the evaluator raises an appropriate error when no labels are present in the graph.
    #     """
    #     del self.graph.nodes['item'].data['label']  # Remove labels for this test
    #     with pytest.raises(ValueError, match="No labeled nodes found in the graph."):
    #         self._evaluator.validate()
