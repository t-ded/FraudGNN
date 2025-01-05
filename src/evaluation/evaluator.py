from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional, cast, Sized

import dgl
import numpy as np
import polars as pl
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from torch import nn, optim
from tqdm import tqdm

from src.data_loading.dynamic_dataset import DynamicDataset
from src.data_loading.graph_builder import GraphDatasetDefinition
from src.data_loading.tabular_dataset import TabularDatasetDefinition

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / 'config/general_config.json'

with open(CONFIG_PATH) as f:
    config = json.load(f)

LOG_DIR = ROOT_DIR / config['logs']

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GNNHyperparameters:
    learning_rate: float
    train_batch_size: int
    validation_batch_size: int
    sampler_fanouts: list[int]


class EvaluationMetricsComputer:

    def __init__(self) -> None:
        self._logits_list: list[torch.Tensor] = []
        self._labels_list: list[torch.Tensor] = []

        self._logits: torch.Tensor = torch.Tensor([])
        self._labels: torch.Tensor = torch.Tensor([])

        self._probabilities: NDArray[np.float32] = np.empty(0, dtype=np.float32)
        self._labels_np: NDArray[np.float32] = np.empty(0, dtype=np.float32)

    def build_logits_labels_tensors_and_numpy_representations(self) -> None:
        self._logits = torch.cat(self._logits_list, dim=0)
        self._labels = torch.cat(self._labels_list, dim=0)

        self._probabilities = self._logits.cpu().numpy()
        self._labels_np = self._labels.cpu().numpy()

    @property
    def num_samples(self) -> int:
        return self._logits.shape[0]

    @property
    def labels_np(self) -> NDArray[np.float32]:
        return self._labels_np

    @property
    def probabilities(self) -> NDArray[np.float32]:
        return self._probabilities

    def total_loss(self, loss_func: nn.Module) -> float:
        return loss_func(self._logits, self._labels)

    @property
    def precision(self) -> float:
        return precision_score(self.labels_np, (self.probabilities > 0.5).astype(np.int_))

    @property
    def recall(self) -> float:
        return recall_score(self.labels_np, (self.probabilities > 0.5).astype(np.int_))

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        self._logits_list.append(logits)
        self._labels_list.append(labels)

    def print_summary(self, loss_func: nn.Module = nn.BCEWithLogitsLoss()) -> None:
        print()
        print('-' * 25)
        print('Evaluation Summary:')
        print(f'- Number of samples: {self.num_samples}')
        print(f'- Precision: {self.precision:.4f}')
        print(f'- Recall: {self.recall:.4f}')
        print(f'- Total {'BCE' if not loss_func else ''} loss: {self.total_loss(loss_func):.4f}')
        print('-' * 25)
        print()


# TODO: Eventually, split Evaluator and Logger (and perhaps even Trainer) into separate classes from which we will compose something like FullPipeline object
class Evaluator:

    def __init__(
            self, model: nn.Module, hyperparameters: GNNHyperparameters,
            tabular_dataset_definition: TabularDatasetDefinition, graph_dataset_definition: GraphDatasetDefinition,
            preprocess_tabular: bool = True, identifier: str = 'GNNEvaluation', save_logs: bool = False,
    ) -> None:
        self._model = model
        self._hyperparameters = hyperparameters
        self._dynamic_dataset = DynamicDataset(
            name=identifier + 'DynamicDataset',
            tabular_dataset_definition=tabular_dataset_definition,
            graph_dataset_definition=graph_dataset_definition,
            preprocess_tabular=preprocess_tabular,
        )
        assert graph_dataset_definition.label_col, 'Cannot perform training and evaluation without label column specified.'
        assert self._dynamic_dataset.graph_dataset.label_node_col is not None, 'Cannot perform training and evaluation without labelled node specified.'
        self._labelled_node_ntype = self._dynamic_dataset.graph_dataset.get_ntype_for_column_name(self._dynamic_dataset.graph_dataset.label_node_col)

        self._criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self._dynamic_dataset.tabular_dataset.imbalance_ratio(graph_dataset_definition.label_col)))
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._hyperparameters.learning_rate)

        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._model.to(self._device)
        assert self._dynamic_dataset.graph is not None, 'Graph should be initialized by this point!'
        self._dynamic_dataset.graph.to(self._device)

        self._save_logs = save_logs
        self._log_dir = Path(LOG_DIR)
        if self._save_logs:
            self._setup_log_dir(hyperparameters, identifier, tabular_dataset_definition.data_path.name)
            self._write_setup_summary(model, hyperparameters, identifier, tabular_dataset_definition, graph_dataset_definition)

        self._validate_sampler_dimensionalities()
        self._sampler = dgl.dataloading.MultiLayerNeighborSampler(self._hyperparameters.sampler_fanouts) # TODO: Generalize this into hyperparam

    def _setup_log_dir(self, hyperparameters: GNNHyperparameters, identifier: str, dataset_name: str) -> None:
        datetime_part = f'_{datetime.today().strftime('%d_%m_%Y_%H_%M')}_'
        hyperparam_part = '_'.join([f'lr{hyperparameters.learning_rate}', f'tbs{hyperparameters.train_batch_size}', f'vbs{hyperparameters.validation_batch_size}'])
        self._log_dir = LOG_DIR / (identifier + '_' + dataset_name + datetime_part + hyperparam_part)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _write_setup_summary(
            self, model: nn.Module, hyperparameters: GNNHyperparameters, identifier: str,
            tabular_dataset_definition: TabularDatasetDefinition, graph_dataset_definition: GraphDatasetDefinition
    ) -> None:

        summary_file = self._log_dir / 'setup_summary.txt'

        train_ratio = tabular_dataset_definition.train_val_test_ratios.train_ratio
        val_ratio = tabular_dataset_definition.train_val_test_ratios.val_ratio
        test_ratio = tabular_dataset_definition.train_val_test_ratios.test_ratio

        node_feature_cols = '\n'.join([f"{node_col}: {feature_cols}" for node_col, feature_cols in graph_dataset_definition.node_feature_cols.items()])
        label_node_col = graph_dataset_definition.label_node_col
        label_col = graph_dataset_definition.label_col
        edge_definitions = '\n'.join([f"{edge_description}: {edge_cols}" for edge_description, edge_cols in graph_dataset_definition.edge_definitions.items()])

        summary_content = f"""
Setup Summary
=============
Identifier: {identifier}
Dataset: {tabular_dataset_definition.data_path.name}
Train/Validation/Test Ratio: {train_ratio}/{val_ratio}/{test_ratio}

Hyperparameters
---------------
Learning Rate: {hyperparameters.learning_rate}
Training Batch Size: {hyperparameters.train_batch_size}
Validation Batch Size: {hyperparameters.validation_batch_size}

Node Feature Columns
----------------
{node_feature_cols}

Node Label Columns
----------------
{label_node_col} - {label_col}

Edge Definitions
----------------
{edge_definitions}

Model Structure
---------------
{model}
"""

        with summary_file.open('w') as sum_file:
            sum_file.write(summary_content.strip())

        logger.info(f'Setup summary written to {summary_file}')

    def _validate_sampler_dimensionalities(self) -> None:
        assert len(self._hyperparameters.sampler_fanouts) == self._model.n_layers, 'There has to be one sampler neighbour count per layer (excluding the last layer)'

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

        training_dataloader = dgl.dataloading.DataLoader(
            self._dynamic_dataset.graph,
            indices={ntype: torch.arange(self._dynamic_dataset.graph.number_of_nodes(ntype)).to(self._device) for ntype in self._dynamic_dataset.graph.ntypes},
            graph_sampler=self._sampler,
            batch_size=self._hyperparameters.train_batch_size,
            shuffle=True,
            drop_last=False,
            device=self._device,
        )

        losses_per_epoch = []

        for epoch in tqdm(range(num_epochs), desc='Training...'):

            epoch_loss = 0

            for input_nodes, output_nodes, blocks in training_dataloader:
                if self._labelled_node_ntype not in output_nodes:
                    continue

                input_features = {ntype: self._dynamic_dataset.graph_features[ntype][indices] for ntype, indices in input_nodes.items()}

                logits = self._model(blocks, input_features)[self._labelled_node_ntype]
                labels = self._dynamic_dataset.graph_dataset.labels[output_nodes[self._labelled_node_ntype]]
                loss = self._criterion(logits, labels)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                epoch_loss += loss.item()

            average_loss = epoch_loss / len(cast(Sized, training_dataloader))
            losses_per_epoch.append(average_loss)
            logger.info(f'Epoch {epoch + 1}/{num_epochs} - Loss: {average_loss}')

        if plot_loss:
            self._plot_loss(losses_per_epoch)

    def _save_plot(self, figure: plt.Figure, name: str) -> None:
        save_path = self._log_dir / f'{name}.png'
        figure.savefig(save_path)
        logger.info(f'Plot saved at {save_path}')

    def _plot_loss(self, losses_per_epoch: list[float]) -> None:
        fig, ax = plt.subplots()
        ax.plot(range(1, len(losses_per_epoch) + 1), losses_per_epoch, marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Over Epochs')
        ax.set_ylim(bottom=0)

        ax.xaxis.set_major_locator(MaxNLocator('auto', integer=True))

        if self._save_logs:
            self._save_plot(fig, 'training_loss')

    def _validate_on_increment(self, incr: pl.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._dynamic_dataset.graph is not None, 'Graph should be initialized by this point!'

        mask = torch.cat((torch.zeros(self._dynamic_dataset.graph.number_of_nodes(self._labelled_node_ntype)), torch.ones(len(incr)))).type(torch.bool)
        self._dynamic_dataset.update_graph_with_increment(incr)

        with torch.no_grad():
            logits = self._model.graph_forward(self._dynamic_dataset.graph, self._dynamic_dataset.graph_features)[self._labelled_node_ntype][mask]
            labels = self._dynamic_dataset.graph_dataset.labels[mask]

        return logits, labels

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

        final_metrics = EvaluationMetricsComputer()
        streaming_batches = self._dynamic_dataset.get_streaming_batches(ldf, self._hyperparameters.validation_batch_size)

        for incr in tqdm(streaming_batches, desc='Evaluating...'):
            final_metrics.update(*self._validate_on_increment(incr.collect()))
        final_metrics.build_logits_labels_tensors_and_numpy_representations()

        self._evaluation_logging(final_metrics, mode, compute_metrics, plot_pr_curve)
        return final_metrics

    def _evaluation_logging(self, final_metrics: EvaluationMetricsComputer, mode: Literal['validation', 'testing'], compute_metrics: bool, plot_pr_curve: bool) -> None:

        logger.info(f'{mode.capitalize()} process finished, final loss: {final_metrics.total_loss(self._criterion):.2f}.')

        results_summary_content = []

        if compute_metrics:
            precision = final_metrics.precision
            recall = final_metrics.recall

            logger.info(f'Final precision on 0.5 threshold: {precision:.2f}.')
            logger.info(f'Final recall on 0.5 threshold: {recall:.2f}.')

            results_summary_content.append(f'{mode.capitalize()} Evaluation Summary\n')
            results_summary_content.append(f'Final Loss: {final_metrics.total_loss(self._criterion):.4f}')
            results_summary_content.append(f'Final Precision (Threshold 0.5): {precision:.4f}')
            results_summary_content.append(f'Final Recall (Threshold 0.5): {recall:.4f}\n')

            if plot_pr_curve:
                self._plot_and_save_pr_curve(final_metrics, mode)
                precision, recall, _ = precision_recall_curve(final_metrics.labels_np, final_metrics.probabilities)
                plt.plot(recall, precision, color='red' if mode == 'testing' else 'orange')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.show()

        if compute_metrics and self._save_logs:
            summary_file = self._log_dir / f'results_summary_{mode}.txt'
            with summary_file.open('w') as results_file:
                results_file.write('\n'.join(results_summary_content).strip())

            logger.info(f'{mode.capitalize()} results summary written to {summary_file}')

    def _plot_and_save_pr_curve(self, final_metrics: EvaluationMetricsComputer, mode: Literal['validation', 'testing']) -> None:
        fig, ax = plt.subplots()
        color = 'red' if mode == 'testing' else 'orange'
        precision, recall, _ = precision_recall_curve(final_metrics.labels_np, final_metrics.probabilities)
        ax.plot(recall, precision, color=color)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        if self._save_logs:
            self._save_plot(fig, f'pr_curve_{mode}')
