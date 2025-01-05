import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from polars.testing import assert_frame_equal

from src.base.build_test_dataset import TEST_DATASET_ALL_COLUMNS, TEST_DATASET_SOURCE_DATA
from src.data_loading.dynamic_dataset import DynamicDataset
from src.data_loading.graph_builder import GraphDatasetDefinition
from src.data_loading.tabular_dataset import TabularDatasetDefinition, TrainValTestRatios

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = ROOT_DIR / 'config/general_config.json'

with open(CONFIG_PATH) as f:
    config = json.load(f)


class TestDynamicDataset:

    _data_path = ROOT_DIR / config['test_data_path'] / 'minimal_sample.csv'

    def setup_method(self) -> None:
        self._dataset = DynamicDataset(
            name='test_dataset',
            tabular_dataset_definition=TabularDatasetDefinition(
                data_path=self._data_path,
                numeric_columns=[], categorical_columns=[], text_columns=[],
                required_columns=TEST_DATASET_ALL_COLUMNS,
                train_val_test_ratios=TrainValTestRatios(0.5, 0.25, 0.25),
            ),
            graph_dataset_definition=GraphDatasetDefinition(
                node_feature_cols={'test_id': ['test_amount'], 'test_customer': []},
                label_node_col=None,
                label_col=None,
                edge_definitions={
                    ('customer', 'makes', 'transaction'): ('test_customer', 'test_id'),
                },
            ),
            preprocess_tabular=False,
        )

    def test_graph_is_setup_correctly(self) -> None:
        test_graph = self._dataset[0]
        assert test_graph is not None
        assert test_graph.num_nodes() == 6
        np.testing.assert_array_equal(test_graph.edges(etype='makes'), (torch.tensor([0, 1, 2]), torch.tensor([0, 1, 2])))
        np.testing.assert_array_equal(self._dataset.graph_features['transaction'].flatten(), torch.tensor([100, 200, 300]))

    def test_get_streaming_batches(self) -> None:
        dataset_type_enum = pl.Enum(['train', 'val', 'test'])
        dataset_types = ['train', 'train', 'train', 'val', 'test']
        for i, incr in enumerate(self._dataset.get_streaming_batches(self._dataset.tabular_dataset.ldf, 1)):
            assert_frame_equal(
                incr,
                pl.LazyFrame(
                    data=(
                        {'row_index': [i]} |
                        {key: [TEST_DATASET_SOURCE_DATA[key][i]] for key in TEST_DATASET_ALL_COLUMNS} |
                        {'train_val_test_mask': pl.Series('train_val_test_mask', [dataset_types[i]], dtype=dataset_type_enum)}
                    ),
                    schema_overrides={'row_index': pl.UInt32},
                ),
            )
        assert self._dataset.num_streaming_batches(self._dataset.tabular_dataset.ldf, 1) == 5
