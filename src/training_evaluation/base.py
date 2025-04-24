from dataclasses import dataclass
from datetime import date
from typing import Literal, NamedTuple

import polars as pl


class TrainTestSplit(NamedTuple):
    from_date: date

    to_date_train_during_hyperparam_opt: date
    to_date_train_during_full: date

    from_date_validation: date
    to_date_validation: date

    from_date_test: date
    to_date_test: date


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
    temporal_strategy: Literal['uniform', 'last']
    smoothing: float
    grad_clip_max_norm: float
    warmup_epochs: int
    weight_decay: float
