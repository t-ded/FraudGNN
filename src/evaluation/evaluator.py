import logging
from dataclasses import dataclass
from typing import Literal

import polars as pl
import torch
from torch import nn, optim
from tqdm import tqdm

from src.data_loading.dynamic_dataset import DynamicDataset
from src.data_loading.graph_builder import GraphDatasetDefinition
from src.data_loading.tabular_dataset import TabularDatasetDefinition

logger = logging.getLogger(__name__)


@dataclass
class GNNHyperparameters:
    learning_rate: float
    batch_size: int


@dataclass
class EvaluationMetrics:
    number_of_samples: int
    total_loss: float


class Evaluator:

    def __init__(
            self, model: nn.Module, hyperparameters: GNNHyperparameters,
            tabular_dataset_definition: TabularDatasetDefinition, graph_dataset_definition: GraphDatasetDefinition,
            preprocess_tabular: bool = True, identifier: str = 'GNNEvaluation',
    ) -> None:
        self._model = model
        self._hyperparameters = hyperparameters
        self._dynamic_dataset = DynamicDataset(
            name=identifier + 'DynamicDataset',
            tabular_dataset_definition=tabular_dataset_definition,
            graph_dataset_definition=graph_dataset_definition,
            preprocess_tabular=preprocess_tabular,
        )
        self._label_nodes = [self._dynamic_dataset.graph_dataset.get_ntype_for_column_name(col) for col in self._dynamic_dataset.graph_dataset.node_label_cols.keys()]

        self._criterion = nn.BCEWithLogitsLoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._hyperparameters.learning_rate)

        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._model.to(self._device)
        assert self._dynamic_dataset.graph is not None, 'Graph should be initialized by this point!'
        self._dynamic_dataset.graph.to(self._device)

    @property
    def hyperparameters(self) -> GNNHyperparameters:
        return self._hyperparameters

    @property
    def dynamic_dataset(self) -> DynamicDataset:
        return self._dynamic_dataset

    def train(self, num_epochs: int) -> None:
        assert self._dynamic_dataset.graph is not None, 'Graph should be initialized by this point!'
        logger.info('Starting the training process...')

        for epoch in tqdm(range(num_epochs), desc='Training...'):
            self._model.train()

            logit_dict = self._model(self._dynamic_dataset.graph, self._dynamic_dataset.graph_features)
            logits = torch.cat([logit_dict[ntype] for ntype in self._label_nodes], dim=0)
            label_dict = self._dynamic_dataset.graph_dataset.labels
            labels = torch.cat([label_dict[ntype] for ntype in self._label_nodes], dim=0).type(torch.float32)
            loss = self._criterion(logits, labels)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            logger.info(f'Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}')

    def _validate_on_increment(self, incr: pl.DataFrame, compute_metrics: bool) -> dict[str, float]:
        assert self._dynamic_dataset.graph is not None, 'Graph should be initialized by this point!'

        metrics: dict[str, float] = {}
        incr_len = len(incr)

        mask = self._get_incr_mask(incr_len)
        self._dynamic_dataset.update_graph_with_increment(incr)

        with torch.no_grad():
            logit_dict = self._model(self._dynamic_dataset.graph, self._dynamic_dataset.graph_features)
            logits = torch.cat([logit_dict[ntype] for ntype in self._label_nodes], dim=0)[mask]
            label_dict = self._dynamic_dataset.graph_dataset.labels
            labels = torch.cat([label_dict[ntype] for ntype in self._label_nodes], dim=0).type(torch.float32)[mask]
            loss = self._criterion(logits, labels)

            if compute_metrics:
                # TODO: Compute evaluation metrics
                pass

        return {'loss': loss.item(), 'n_labelled_samples': incr_len} | metrics

    def _get_incr_mask(self, incr_len: int) -> torch.Tensor:
        assert self._dynamic_dataset.graph is not None, 'Graph should be initialized by this point!'

        return torch.cat((
            *(torch.zeros(self._dynamic_dataset.graph.num_nodes(label_node)) for label_node in self._label_nodes),
            torch.ones(incr_len),
        )).type(torch.bool)

    def stream_evaluate(self, mode: Literal['validation', 'testing']) -> EvaluationMetrics:
        assert self._dynamic_dataset.graph is not None, 'Graph should be initialized by this point!'

        if mode == 'validation':
            logger.info('Starting the validation process...')
            compute_metrics = False
            ldf = self._dynamic_dataset.tabular_dataset.val_ldf
        elif mode == 'testing':
            logger.info('Starting the testing process...')
            compute_metrics = True
            ldf = self._dynamic_dataset.tabular_dataset.test_ldf
        else:
            raise ValueError('Invalid mode of evaluation.')

        self._model.eval()

        final_metrics = EvaluationMetrics(number_of_samples=0, total_loss=0.0)
        streaming_batches = self._dynamic_dataset.get_streaming_batches(ldf, self._hyperparameters.batch_size)

        for incr in tqdm(streaming_batches, desc='Evaluating...'):
            metrics = self._validate_on_increment(incr.collect(), compute_metrics)
            final_metrics.total_loss += metrics['loss']
            final_metrics.number_of_samples += int(metrics['n_labelled_samples'])

        if mode == 'validation':
            logger.info(f'Validation process finished, final validation loss: {final_metrics.total_loss / final_metrics.number_of_samples:_.2f}.')
        elif mode == 'testing':
            logger.info(f'Testing process finished, final validation loss: {final_metrics.total_loss / final_metrics.number_of_samples:_.2f}.')

        # TODO: More profound metric return
        return final_metrics
