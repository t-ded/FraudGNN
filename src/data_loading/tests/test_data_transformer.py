import json
from pathlib import Path

from src.base.build_test_dataset import TEST_DATASET_ALL_COLUMNS, TEST_DATASET_NUMERIC_COLUMNS, TEST_DATASET_CATEGORICAL_COLUMNS, TEST_DATASET_TEXT_COLUMNS
from src.data_loading.data_transformer import DataTransformer
from src.data_loading.tabular_dataset import TabularDataset, TabularDatasetDefinition

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = ROOT_DIR / 'config/general_config.json'

with open(CONFIG_PATH) as f:
    config = json.load(f)


class TestDataTransformer:

    _data_path = ROOT_DIR / config['test_data_path'] / 'minimal_sample.csv'

    def setup_method(self) -> None:
        self._dataset = TabularDataset(
            TabularDatasetDefinition(
                data_path=self._data_path,
                numeric_columns=TEST_DATASET_NUMERIC_COLUMNS, categorical_columns=TEST_DATASET_CATEGORICAL_COLUMNS, text_columns=TEST_DATASET_TEXT_COLUMNS,
                required_columns=TEST_DATASET_ALL_COLUMNS,
                train_val_test_portions=(0.7, 0.15, 0.15)
            ),
        )
        self._transformer = DataTransformer()

    def test_numeric_normalization(self) -> None:
        self._transformer.normalize_numeric(self._dataset)
        assert self._dataset.df.select('test_amount').to_series().to_list() == [0, 0.25, 0.5, 0.75, 1]

    def test_categorical_encoding(self) -> None:
        self._transformer.encode_categorical(self._dataset)
        assert self._dataset.df.select('test_category').to_series().to_list() == [0, 1, 2, 3, 4]
