from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, kw_only=True)
class TrainTestSplitSettings:
    from_date: pl.Date
    to_date: pl.Date
    label_safety_gap_days: int
    eval_set_len_days: int


@dataclass(kw_only=True)
class GNNTrainingHyperparameters:
    learning_rate: float
    train_batch_size: int
    validation_batch_size: int
    num_train_epochs: int
    positive_class_weight: float
    sampler_fanouts: list[int]
    smoothing: float
    grad_norm_clipping: float
    warmup_epochs: int
    train_test_split_settings: TrainTestSplitSettings
