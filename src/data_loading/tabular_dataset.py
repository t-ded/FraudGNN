from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple

import polars as pl
from polars._typing import IntoExpr


class TrainValTestRatios(NamedTuple):
    train_ratio: float
    val_ratio: float
    test_ratio: float


@dataclass
class TabularDatasetDefinition:
    data_path: Path
    numeric_columns: list[str]
    categorical_columns: list[str]
    text_columns: list[str]
    required_columns: list[str]
    train_val_test_ratios: TrainValTestRatios

class TabularDataset:

    def __init__(self, tabular_dataset_definition: TabularDatasetDefinition) -> None:

        self._numeric_columns = tabular_dataset_definition.numeric_columns
        self._categorical_columns = tabular_dataset_definition.categorical_columns
        self._text_columns = tabular_dataset_definition.text_columns
        self._validate_column_type_assignment()

        self._full_ldf = pl.scan_csv(tabular_dataset_definition.data_path)
        self._ldf = pl.scan_csv(tabular_dataset_definition.data_path).select(tabular_dataset_definition.required_columns)

        self._train_val_test_ratios = tabular_dataset_definition.train_val_test_ratios
        self._create_train_val_test_mask()

    def _validate_column_type_assignment(self) -> None:
        assert not set(self._numeric_columns) & set(self._text_columns), f'Numeric and text columns have joint elements: {set(self._numeric_columns) & set(self._text_columns)}.'
        assert not set(self._numeric_columns) & set(self._categorical_columns), f'Numeric and categorical columns have joint elements: {set(self._numeric_columns) & set(self._categorical_columns)}.'
        assert not set(self._categorical_columns) & set(self._text_columns), f'Categorical and text columns have joint elements: {set(self._categorical_columns) & set(self._text_columns)}.'

    def _create_train_val_test_mask(self) -> None:
        self._validate_train_val_test_split_ratios()
        self._ldf = self._ldf.with_row_index('row_index')

        df_len = self._ldf.select(pl.len()).collect().item()
        dataset_split = pl.Enum(['train', 'val', 'test'])
        self.with_columns(
            pl.when(pl.col('row_index') <= df_len * (1 - self._train_val_test_ratios.test_ratio - self._train_val_test_ratios.val_ratio))
            .then(pl.lit('train'))
            .when(pl.col('row_index') <= df_len * (1 - self._train_val_test_ratios.test_ratio))
            .then(pl.lit('val'))
            .otherwise(pl.lit('test'))
            .cast(dataset_split)
            .alias('train_val_test_mask')
        )

    def _validate_train_val_test_split_ratios(self) -> None:
        assert sum(self._train_val_test_ratios) == 1.0

    @property
    def columns(self) -> list[str]:
        return self._numeric_columns + self._categorical_columns + self._text_columns

    @property
    def numeric_columns(self) -> list[str]:
        return self._numeric_columns

    @property
    def categorical_columns(self) -> list[str]:
        return self._categorical_columns

    @property
    def text_columns(self) -> list[str]:
        return self._text_columns

    @property
    def ldf(self) -> pl.LazyFrame:
        return self._ldf

    @property
    def df(self) -> pl.DataFrame:
        return self._ldf.collect()

    @property
    def train_ldf(self) -> pl.LazyFrame:
        return self._ldf.filter((pl.col('train_val_test_mask') == 'train'))

    @property
    def val_ldf(self) -> pl.LazyFrame:
        return self._ldf.filter((pl.col('train_val_test_mask') == 'val'))

    @property
    def test_ldf(self) -> pl.LazyFrame:
        return self._ldf.filter((pl.col('train_val_test_mask') == 'test'))

    def imbalance_ratio(self, col: str) -> float:
        grouping = self.train_ldf.group_by(col).agg(pl.len()).collect()
        return grouping.filter((pl.col(col).cast(pl.Boolean).not_())).select(pl.col('len')).item() / grouping.filter((pl.col(col).cast(pl.Boolean))).select(pl.col('len')).item()

    def with_columns(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> None:
        self._ldf = self._ldf.with_columns(*exprs)
