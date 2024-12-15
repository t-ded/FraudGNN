import polars as pl
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from src.data_loading.dataset import Dataset


class DataTransformer:

    def __init__(self) -> None:
        self._scaler = MinMaxScaler()

    def normalize_numeric(self, dataset: Dataset) -> None:
        normalize_expressions: list[pl.Series] = []
        for column in dataset.numeric_columns:
            inp = dataset.ldf.select(column).collect()
            normalize_expressions.append(pl.Series(column, self._scaler.fit_transform(inp).flatten()))
        dataset.with_columns(normalize_expressions)

    @staticmethod
    def encode_categorical(dataset: Dataset) -> None:
        encoded_expressions: list[pl.Series] = []
        for column in dataset.categorical_columns:
            encoder = LabelEncoder()
            inp = dataset.ldf.select(column).collect().to_numpy().flatten()
            encoded_expressions.append(pl.Series(column, encoder.fit_transform(inp)))
        dataset.with_columns(encoded_expressions)

    @staticmethod
    def encode_text(dataset: Dataset) -> None:
        raise NotImplementedError('Text data transformation not implemented yet.')
