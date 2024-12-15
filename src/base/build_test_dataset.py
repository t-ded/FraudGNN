import json
from pathlib import Path

import polars as pl

TEST_DATASET_SOURCE_DATA = {
    'test_id': [1, 2, 3, 4, 5],
    'test_amount': [100, 200, 300, 400, 500],
    'test_customer': ['A', 'B', 'C', 'D', 'E'],
    'test_counterparty': ['aaa', 'bbb', 'ccc', 'ddd', 'eee'],
    'test_category': ['cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5'],
}

TEST_DATASET_ALL_COLUMNS: list[str] = list(TEST_DATASET_SOURCE_DATA.keys())
TEST_DATASET_NUMERIC_COLUMNS: list[str] = [
    'test_amount',
]
TEST_DATASET_CATEGORICAL_COLUMNS: list[str] = [
    'test_category',
]
TEST_DATASET_TEXT_COLUMNS: list[str] = [

]

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = ROOT_DIR / 'config/general_config.json'

with open(CONFIG_PATH) as f:
    config = json.load(f)

if __name__ == '__main__':
    pl.DataFrame(TEST_DATASET_SOURCE_DATA).write_csv(ROOT_DIR / config['test_data_path'] / 'minimal_sample.csv')
