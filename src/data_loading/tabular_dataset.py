from dataclasses import dataclass
from typing import Iterable

import polars as pl
from polars._typing import IntoExpr


@dataclass
class TabularDatasetDefinition:
    data_path: str
    numeric_columns: list[str]
    categorical_columns: list[str]
    text_columns: list[str]
    required_columns: list[str]
    train_val_test_portions: tuple[float, float, float]


class TabularDataset:

    def __init__(self, tabular_dataset_definition: TabularDatasetDefinition) -> None:

        self._numeric_columns = tabular_dataset_definition.numeric_columns
        self._categorical_columns = tabular_dataset_definition.categorical_columns
        self._text_columns = tabular_dataset_definition.text_columns

        self._full_ldf = pl.scan_csv(tabular_dataset_definition.data_path)
        self._ldf = pl.scan_csv(tabular_dataset_definition.data_path).select(tabular_dataset_definition.required_columns)

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

    def with_columns(self, *exprs: IntoExpr | Iterable[IntoExpr]) -> None:
        self._ldf = self._ldf.with_columns(*exprs)
