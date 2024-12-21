import logging
from dataclasses import dataclass
from typing import Literal

import polars as pl
import torch
from torch import nn, optim
from tqdm import tqdm

from src.data_loading.dynamic_dataset import DynamicDataset

logger = logging.getLogger(__name__)


@dataclass
class GNNHyperparameters:
    learning_rate: float
    batch_size: int


class Evaluator:

    def __init__(self, model: nn.Module, hyperparameters: GNNHyperparameters, dynamic_dataset: DynamicDataset) -> None:
        self._model = model
        self._hyperparameters = hyperparameters
        self._dynamic_dataset = dynamic_dataset

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
        metrics: dict[str, float] = {}

        self._dynamic_dataset.update_graph_with_increment(incr)
        with torch.no_grad():
            label_col = 'transaction'  # TODO: Generalize this
            logits = self._model(self._dynamic_dataset.graph, self._dynamic_dataset.graph_features)[label_col]
            labels = self._dynamic_dataset.graph_dataset.labels[label_col].type(torch.float32)
            loss = self._criterion(logits, labels)

            if compute_metrics:
                # TODO: Compute evaluation metrics
                pass

        return {'loss': loss.item(), 'n_labelled_samples': len(labels)} | metrics

    def stream_evaluate(self, mode: Literal['validation', 'testing']) -> None:
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

        total_loss = 0.0
        total_samples = 0
        streaming_batches = self._dynamic_dataset.get_streaming_batches(ldf, self._hyperparameters.batch_size)

        for incr in tqdm(streaming_batches, desc='Evaluating'):
            metrics = self._validate_on_increment(incr.collect(), compute_metrics)
            total_loss += metrics['loss']
            total_samples += int(metrics['n_labelled_samples'])

        if mode == 'validation':
            logger.info(f'Validation process finished, final validation loss: {total_loss / total_samples:_2f}.')
        elif mode == 'testing':
            logger.info(f'Testing process finished, final validation loss: {total_loss / total_samples:_2f}.')
