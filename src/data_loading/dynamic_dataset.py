import logging
from typing import Generator

import polars as pl
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
