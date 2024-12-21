import json
from pathlib import Path
from typing import cast

from src.base.build_test_dataset import TEST_DATASET_ALL_COLUMNS, TEST_DATASET_NUMERIC_COLUMNS, TEST_DATASET_CATEGORICAL_COLUMNS, TEST_DATASET_TEXT_COLUMNS, TEST_DATASET_SOURCE_DATA
from src.data_loading.tabular_dataset import TabularDataset, TabularDatasetDefinition, TrainValTestRatios

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = ROOT_DIR / 'config/general_config.json'

with open(CONFIG_PATH) as f:
    config = json.load(f)


class TestTabularDataset:

    _data_path = ROOT_DIR / config['test_data_path'] / 'minimal_sample.csv'

    def test_data_loading(self) -> None:
        tabular_dataset = TabularDataset(
            TabularDatasetDefinition(
                data_path=self._data_path,
                numeric_columns=[], categorical_columns=[], text_columns=[],
                required_columns=TEST_DATASET_ALL_COLUMNS,
                train_val_test_ratios=TrainValTestRatios(0.5, 0.25, 0.25)
            ),
        )
        assert tabular_dataset.df.shape == (cast(list, TEST_DATASET_SOURCE_DATA['test_id'])[-1], len(TEST_DATASET_ALL_COLUMNS) + 2)
        assert tabular_dataset.df.select('train_val_test_mask').to_series().to_list() == ['train', 'train', 'train', 'val', 'test']

    def test_data_loading_with_categories(self) -> None:
        tabular_dataset = TabularDataset(
            TabularDatasetDefinition(
                data_path=self._data_path,
                numeric_columns=TEST_DATASET_NUMERIC_COLUMNS, categorical_columns=TEST_DATASET_CATEGORICAL_COLUMNS, text_columns=TEST_DATASET_TEXT_COLUMNS,
                required_columns=TEST_DATASET_ALL_COLUMNS,
                train_val_test_ratios=TrainValTestRatios(0.7, 0.15, 0.15)
            ),
        )
        assert tabular_dataset.numeric_columns == TEST_DATASET_NUMERIC_COLUMNS
        assert tabular_dataset.categorical_columns == TEST_DATASET_CATEGORICAL_COLUMNS
        assert tabular_dataset.text_columns == TEST_DATASET_TEXT_COLUMNS

    def test_subsample_columns(self) -> None:
        tabular_dataset = TabularDataset(
            TabularDatasetDefinition(
                data_path=self._data_path,
                numeric_columns=[], categorical_columns=[], text_columns=[],
                required_columns=['test_id'],
                train_val_test_ratios=TrainValTestRatios(0.7, 0.15, 0.15)
            ),
        )
        assert tabular_dataset.df.shape == (5, 3)
