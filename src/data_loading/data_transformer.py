import polars as pl
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

from src.data_loading.tabular_dataset import TabularDataset


class DataTransformer:

    def __init__(self) -> None:
        self._scaler = StandardScaler()
        self._encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, encoded_missing_value=-1)

    def fit_numeric_scaler(self, df: pl.DataFrame, numeric_columns: list[str]) -> None:
        self._scaler.fit(df.select(numeric_columns))

    def get_normalized_numeric_columns(self, df: pl.DataFrame, numeric_columns: list[str]) -> list[pl.Series]:
        transformed_values = self._scaler.transform(df.select(numeric_columns))
        return [pl.Series(name, values) for name, values in zip(numeric_columns, transformed_values.T)]

    def fit_encoder(self, df: pl.DataFrame, categorical_columns: list[str]) -> None:
        self._encoder.fit(df.select(categorical_columns))

    def get_encoded_categorical_columns(self, df: pl.DataFrame, categorical_columns: list[str]) -> list[pl.Series]:
        transformed_values = self._encoder.transform(df.select(categorical_columns))
        return [pl.Series(name, values) for name, values in zip(categorical_columns, transformed_values.T)]

    @staticmethod
    def encode_text(tabular_dataset: TabularDataset) -> None:
        raise NotImplementedError('Text data transformation not implemented yet.')
