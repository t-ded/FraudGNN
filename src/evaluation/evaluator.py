from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import polars as pl
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from torch import nn, optim
from tqdm import tqdm

from src.data_loading.dynamic_dataset import DynamicDataset
from src.data_loading.graph_builder import GraphDatasetDefinition
from src.data_loading.tabular_dataset import TabularDatasetDefinition

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GNNHyperparameters:
    learning_rate: float
    batch_size: int


class EvaluationMetricsComputer:

    def __init__(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        self._logits = logits
        self._labels = labels

        self._probabilities: Optional[NDArray[np.float32]] = None
        self._labels_np: Optional[NDArray[np.float32]] = None

    @property
    def num_samples(self) -> int:
        return self._logits.shape[0]

    @property
    def labels_np(self) -> NDArray[np.float32]:
        if self._labels_np is None:
            self._labels_np = self._labels.cpu().numpy()
        return self._labels_np

    @property
    def probabilities(self) -> NDArray[np.float32]:
        if self._probabilities is None:
            self._probabilities = self._logits.cpu().numpy()
        return self._probabilities

    def total_loss(self, loss_func: nn.Module) -> float:
        return loss_func(self._logits, self._labels)

    @property
    def precision(self) -> float:
        return precision_score(self.labels_np, (self.probabilities > 0.5).astype(np.int_))

    @property
    def recall(self) -> float:
        return recall_score(self.labels_np, (self.probabilities > 0.5).astype(np.int_))

    def update(self, other: EvaluationMetricsComputer) -> None:
        self._logits = torch.cat((other._logits, self._logits), dim=0)
        self._labels = torch.cat((other._labels, self._labels), dim=0)

    def print_summary(self, loss_func: nn.Module = nn.BCEWithLogitsLoss()) -> None:
        print()
        print('-' * 25)
        print(f'Evaluation Summary:')
        print(f'- Number of samples: {self.num_samples}')
        print(f'- Precision: {self.precision:.4f}')
        print(f'- Recall: {self.recall:.4f}')
        print(f'- Total {'BCE' if not loss_func else ''} loss: {self.total_loss(loss_func):.4f}')
        print('-' * 25)
        print()


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

        self._criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [self._dynamic_dataset.tabular_dataset.imbalance_ratio(col)
                 for col in graph_dataset_definition.node_label_cols.values()]
            ).mean()
        )
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

    @property
    def criterion(self) -> nn.BCEWithLogitsLoss:
        return self._criterion

    def train(self, num_epochs: int, plot_loss: bool = False) -> None:
        assert self._dynamic_dataset.graph is not None, 'Graph should be initialized by this point!'
        logger.info('Starting the training process...')

        losses_per_epoch = []

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

            losses_per_epoch.append(loss.item())
            logger.info(f'\nEpoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}')

        if plot_loss:
            plt.plot(range(1, num_epochs + 1), losses_per_epoch, marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Epochs')
            plt.show()

    def _validate_on_increment(self, incr: pl.DataFrame) -> EvaluationMetricsComputer:
        # TODO: For now, assume only transactions will be labelled -> unique node per row of incr, i.e., whole incr is validation set, number of samples is equal to incr length etc.
        assert self._dynamic_dataset.graph is not None, 'Graph should be initialized by this point!'

        mask: dict[str, torch.Tensor] = self._get_incr_mask(incr)
        self._dynamic_dataset.update_graph_with_increment(incr)

        with torch.no_grad():
            logit_dict = self._model(self._dynamic_dataset.graph, self._dynamic_dataset.graph_features)
            logits = torch.cat([logit_dict[ntype][mask[ntype]] for ntype in self._label_nodes], dim=0)
            label_dict = self._dynamic_dataset.graph_dataset.labels
            labels = torch.cat([label_dict[ntype][mask[ntype]]for ntype in self._label_nodes], dim=0).type(torch.float32)

        return EvaluationMetricsComputer(logits=logits, labels=labels)

    def _get_incr_mask(self, incr: pl.DataFrame) -> dict[str, torch.Tensor]:
        assert self._dynamic_dataset.graph is not None, 'Graph should be initialized by this point!'

        mask = {}
        for label_node in self._label_nodes:
            ntype_num_nodes = self._dynamic_dataset.graph.number_of_nodes(label_node)
            mask[label_node] = torch.cat((torch.zeros(ntype_num_nodes), torch.ones(len(incr)))).type(torch.bool)
        return mask

    def stream_evaluate(self, mode: Literal['validation', 'testing'], compute_metrics: Optional[bool] = None, plot_pr_curve: bool = False) -> EvaluationMetricsComputer:
        assert self._dynamic_dataset.graph is not None, 'Graph should be initialized by this point!'

        if mode == 'validation':
            logger.info('Starting the validation process...')
            if compute_metrics is None:
                compute_metrics = False
            ldf = self._dynamic_dataset.tabular_dataset.val_ldf
        elif mode == 'testing':
            logger.info('Starting the testing process...')
            if compute_metrics is None:
                compute_metrics = True
            ldf = self._dynamic_dataset.tabular_dataset.test_ldf
        else:
            raise ValueError('Invalid mode of evaluation.')

        self._model.eval()

        final_metrics = EvaluationMetricsComputer(logits=torch.Tensor([]), labels=torch.Tensor([]))
        streaming_batches = self._dynamic_dataset.get_streaming_batches(ldf, self._hyperparameters.batch_size)

        for incr in tqdm(streaming_batches, desc='Evaluating...'):
            final_metrics.update(self._validate_on_increment(incr.collect()))

        logger.info(f'{mode.capitalize()} process finished, final loss: {final_metrics.total_loss(self._criterion):.2f}.')

        if compute_metrics:
            logger.info(f'Final precision on 0.5 threshold: {final_metrics.precision:.2f}.')
            logger.info(f'Final recall on 0.5 threshold: {final_metrics.recall:.2f}.')

            if plot_pr_curve:
                precision, recall, _ = precision_recall_curve(final_metrics.labels_np, final_metrics.probabilities)
                plt.plot(recall, precision, color='red' if mode == 'testing' else 'orange')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.show()

        return final_metrics
