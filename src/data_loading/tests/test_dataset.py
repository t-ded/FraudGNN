import json

from src.data_loading.dataset import Dataset


with open('../../../config/general_config.json') as f:
    config = json.load(f)


class TestDataset:

    _data_path = '../../../' + config['test_data_path'] + '/minimal_sample.csv'
    _all_columns = ['test_id', 'test_amount', 'test_category']

    def test_data_loading(self) -> None:
        dataset = Dataset(
            data_path=self._data_path,
            numeric_columns=[], categorical_columns=[], text_columns=[],
            required_columns=self._all_columns,
        )
        assert dataset.df.shape == (5, 3)

    def test_data_loading_with_categories(self) -> None:
        dataset = Dataset(
            data_path=self._data_path,
            numeric_columns=['test_amount'], categorical_columns=['test_category'], text_columns=[],
            required_columns=self._all_columns,
        )
        assert dataset.numeric_columns == ['test_amount']
        assert dataset.categorical_columns == ['test_category']

    def test_subsample_columns(self) -> None:
        dataset = Dataset(
            data_path=self._data_path,
            numeric_columns=[], categorical_columns=[], text_columns=[],
            required_columns=['test_id']
        )
        assert dataset.df.shape == (5, 1)
