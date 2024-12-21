import logging
from functools import cached_property
from typing import Generator, Optional

import polars as pl
import torch
from dgl import DGLGraph
from dgl.data import DGLDataset

from src.data_loading.graph_builder import GraphDataset, GraphDatasetDefinition
from src.data_loading.tabular_dataset import TabularDataset, TabularDatasetDefinition

logger = logging.getLogger(__name__)


class DynamicDataset(DGLDataset):

    def __init__(self, name: str, tabular_dataset_definition: TabularDatasetDefinition, graph_dataset_definition: GraphDatasetDefinition, verbose: bool = True) -> None:
        self._tabular_dataset = TabularDataset(tabular_dataset_definition)
        self._graph_dataset = GraphDataset(graph_dataset_definition)
        super().__init__(name=name, verbose=verbose)

    def process(self) -> None:
        self._graph_dataset.build_graph(self._tabular_dataset.train_ldf)

    def __getitem__(self, idx: int) -> DGLGraph:
        return self._graph_dataset.graph

    def __len__(self) -> int:
        return 1

    @property
    def tabular_dataset(self) -> TabularDataset:
        return self._tabular_dataset

    @property
    def graph_dataset(self) -> GraphDataset:
        return self._graph_dataset

    @property
    def graph(self) -> Optional[DGLGraph]:
        return self._graph_dataset.graph

    @cached_property
    def graph_features(self) -> dict[str, torch.Tensor]:
        features = {}
        if self._graph_dataset.graph is not None:
            for feature, ntype_feature_values in self._graph_dataset.graph.ndata.items():
                if feature not in self._graph_dataset.node_label_cols.values():
                    for ntype, feature_values in ntype_feature_values.items():
                        if ntype not in features:
                            features[ntype] = feature_values.type(torch.float32)
                        else:
                            features[ntype] = torch.cat((features[ntype], feature_values.type(torch.float32)), dim=1)
        return features

    @staticmethod
    def get_streaming_batches(frame: pl.LazyFrame, batch_size: int) -> Generator[pl.LazyFrame, None, None]:
        assert 'row_index' in frame.collect_schema().names(), 'Dataset will be split into streaming batches based on given row index, however, given frame does not contain row index.'
        min_idx: int = frame.select('row_index').min().collect().item()
        max_idx: int = frame.select('row_index').max().collect().item()
        for idx in range(min_idx, max_idx + 1, batch_size):
            yield frame.filter((pl.col('row_index').is_between(idx, idx + batch_size - 1)))

    @staticmethod
    def num_streaming_batches(frame: pl.LazyFrame, batch_size: int) -> int:
        row_count = frame.select(pl.len()).collect().item()
        return (row_count + batch_size - 1) // batch_size

    def update_graph_with_increment(self, incr: pl.DataFrame) -> None:
        self._graph_dataset.update_graph(incr)
