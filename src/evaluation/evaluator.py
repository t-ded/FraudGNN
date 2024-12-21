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
            identifier: str = 'GNNEvaluation',
    ) -> None:
        self._model = model
        self._hyperparameters = hyperparameters
        self._dynamic_dataset = DynamicDataset(name=identifier + 'DynamicDataset', tabular_dataset_definition=tabular_dataset_definition, graph_dataset_definition=graph_dataset_definition)

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

        for epoch in range(num_epochs):
            self._model.train()

            label_col = 'transaction'  # TODO: Generalize this
            logits = self._model(self._dynamic_dataset.graph, self._dynamic_dataset.graph_features)[label_col]
            labels = self._dynamic_dataset.graph_dataset.labels[label_col].type(torch.float32)
            loss = self._criterion(logits, labels)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            logger.info(f'Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}')

    def _validate_on_increment(self, incr: pl.DataFrame, compute_metrics: bool) -> dict[str, float]:
        assert self._dynamic_dataset.graph is not None, 'Graph should be initialized by this point!'

        metrics: dict[str, float] = {}

        label_col = 'transaction'  # TODO: Generalize this
        mask = torch.cat((
            torch.zeros(self._dynamic_dataset.graph.num_nodes(label_col)),
            torch.ones(len(incr)),
        )).type(torch.bool)
        self._dynamic_dataset.update_graph_with_increment(incr)

        with torch.no_grad():
            logits = self._model(self._dynamic_dataset.graph, self._dynamic_dataset.graph_features)[label_col][mask]
            labels = self._dynamic_dataset.graph_dataset.labels[label_col].type(torch.float32)[mask]
            loss = self._criterion(logits, labels)

            if compute_metrics:
                # TODO: Compute evaluation metrics
                pass

        return {'loss': loss.item(), 'n_labelled_samples': len(labels)} | metrics

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

        for incr in tqdm(streaming_batches, desc='Evaluating'):
            metrics = self._validate_on_increment(incr.collect(), compute_metrics)
            final_metrics.total_loss += metrics['loss']
            final_metrics.number_of_samples += int(metrics['n_labelled_samples'])

        if mode == 'validation':
            logger.info(f'Validation process finished, final validation loss: {final_metrics.total_loss / final_metrics.number_of_samples:_.2f}.')
        elif mode == 'testing':
            logger.info(f'Testing process finished, final validation loss: {final_metrics.total_loss / final_metrics.number_of_samples:_.2f}.')

        # TODO: More profound metric return
        return final_metrics
