import polars as pl
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from src.data_loading.tabular_dataset import TabularDataset


class DataTransformer:

    def __init__(self) -> None:
        self._scaler = MinMaxScaler()

    def normalize_numeric(self, tabular_dataset: TabularDataset) -> None:
        normalize_expressions: list[pl.Series] = []
        for column in tabular_dataset.numeric_columns:
            inp = tabular_dataset.ldf.select(column).collect()
            normalize_expressions.append(pl.Series(column, self._scaler.fit_transform(inp).flatten()))
        tabular_dataset.with_columns(normalize_expressions)

    @staticmethod
    def encode_categorical(tabular_dataset: TabularDataset) -> None:
        encoded_expressions: list[pl.Series] = []
        for column in tabular_dataset.categorical_columns:
            encoder = LabelEncoder()
            inp = tabular_dataset.ldf.select(column).collect().to_numpy().flatten()
            encoded_expressions.append(pl.Series(column, encoder.fit_transform(inp)))
        tabular_dataset.with_columns(encoded_expressions)

    @staticmethod
    def encode_text(tabular_dataset: TabularDataset) -> None:
        raise NotImplementedError('Text data transformation not implemented yet.')
