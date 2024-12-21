import json
from pathlib import Path

import numpy as np

from src.base.build_test_dataset import TEST_DATASET_ALL_COLUMNS, TEST_DATASET_NUMERIC_COLUMNS, TEST_DATASET_CATEGORICAL_COLUMNS, TEST_DATASET_TEXT_COLUMNS
from src.data_loading.data_transformer import DataTransformer
from src.data_loading.tabular_dataset import TabularDataset, TabularDatasetDefinition, TrainValTestRatios

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
                train_val_test_ratios=TrainValTestRatios(0.7, 0.15, 0.15)
            ),
        )
        self._transformer = DataTransformer()

    def test_numeric_normalization(self) -> None:
        self._transformer.fit_numeric_scaler(self._dataset.df, TEST_DATASET_NUMERIC_COLUMNS)
        self._dataset.with_columns(self._transformer.get_normalized_numeric_columns(self._dataset.df, TEST_DATASET_NUMERIC_COLUMNS))
        np.testing.assert_array_almost_equal(self._dataset.df.select('test_amount').to_series().to_list(), [-1.414, -0.707, 0.0, 0.707, 1.414], decimal=3)

    def test_numeric_normalization_different_fit_transform(self) -> None:
        self._transformer.fit_numeric_scaler(self._dataset.train_ldf.collect(), TEST_DATASET_NUMERIC_COLUMNS)
        self._dataset.with_columns(self._transformer.get_normalized_numeric_columns(self._dataset.df, TEST_DATASET_NUMERIC_COLUMNS))
        np.testing.assert_array_almost_equal(self._dataset.df.select('test_amount').to_series().to_list(), [-1.342, -0.447, 0.447, 1.342, 2.236], decimal=3)

    def test_categorical_encoding(self) -> None:
        self._transformer.fit_encoder(self._dataset.df, TEST_DATASET_CATEGORICAL_COLUMNS)
        self._dataset.with_columns(self._transformer.get_encoded_categorical_columns(self._dataset.df, TEST_DATASET_CATEGORICAL_COLUMNS))
        assert self._dataset.df.select('test_category').to_series().to_list() == [0, 1, 2, 3, 4]

    def test_categorical_encoding_different_fit_transform(self) -> None:
        self._transformer.fit_encoder(self._dataset.train_ldf.collect(), TEST_DATASET_CATEGORICAL_COLUMNS)
        self._dataset.with_columns(self._transformer.get_encoded_categorical_columns(self._dataset.df, TEST_DATASET_CATEGORICAL_COLUMNS))
        assert self._dataset.df.select('test_category').to_series().to_list() == [0, 1, 2, 3, -1]
