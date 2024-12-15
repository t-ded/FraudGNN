from typing import Iterable

import polars as pl
from polars._typing import IntoExpr


class TabularDataset:

    def __init__(self, data_path: str, numeric_columns: list[str], categorical_columns: list[str], text_columns: list[str], required_columns: list[str]) -> None:

        self._numeric_columns = numeric_columns
        self._categorical_columns = categorical_columns
        self._text_columns = text_columns

        self._full_ldf = pl.scan_csv(data_path)
        self._ldf = pl.scan_csv(data_path).select(required_columns)

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
