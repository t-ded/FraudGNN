import json
from pathlib import Path

from src.data_loading.tabular_dataset import TabularDataset

ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = ROOT_DIR / 'config/general_config.json'

with open(CONFIG_PATH) as f:
    config = json.load(f)


class TestTabularDataset:

    _data_path = ROOT_DIR / config['test_data_path'] / 'minimal_sample.csv'
    _all_columns = ['test_id', 'test_amount', 'test_category']

    def test_data_loading(self) -> None:
        tabular_dataset = TabularDataset(
            data_path=self._data_path,
            numeric_columns=[], categorical_columns=[], text_columns=[],
            required_columns=self._all_columns,
        )
        assert tabular_dataset.df.shape == (5, 3)

    def test_data_loading_with_categories(self) -> None:
        tabular_dataset = TabularDataset(
            data_path=self._data_path,
            numeric_columns=['test_amount'], categorical_columns=['test_category'], text_columns=[],
            required_columns=self._all_columns,
        )
        assert tabular_dataset.numeric_columns == ['test_amount']
        assert tabular_dataset.categorical_columns == ['test_category']

    def test_subsample_columns(self) -> None:
        tabular_dataset = TabularDataset(
            data_path=self._data_path,
            numeric_columns=[], categorical_columns=[], text_columns=[],
            required_columns=['test_id']
        )
        assert tabular_dataset.df.shape == (5, 1)
