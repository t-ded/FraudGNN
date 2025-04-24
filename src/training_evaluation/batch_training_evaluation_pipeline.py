from datetime import date
from typing import Optional, Any, cast

import polars as pl
import torch
from polars._typing import IntoExprColumn
from torch_geometric.data import HeteroData

from src.base.base import NodeType, PrimaryNodeDefinition, AssociatedFeatureDefinition, PREPROCESSING_OPTIONS, EdgeDescription, MonoPartiteEdgeType, BiPartiteEdgeType
from src.base.graph_dataset_definition import GraphDatasetDefinition
from src.training_evaluation.base import TrainTestSplitSettings, TrainTestSplit


class BatchTrainingEvaluationPipeline:

    def __init__(
            self,
            input_data_path: str,
            label_col: pl.Expr,
            # model: nn.Module,
            # training_hyperparameters: GNNTrainingHyperparameters,
            train_test_split_settings: TrainTestSplitSettings,
            graph_dataset_definition: GraphDatasetDefinition,
            # identifier: str = 'GNNBatchTrainingEvaluationPipeline',
            # logger_settings: LoggerSettings,
            dataset_filter_exprs: Optional[list[IntoExprColumn]] = None,
            inference_filter_exprs: Optional[list[IntoExprColumn]] = None,
            timestamp_col: str = 'timestamp',
            cust_identifier: str = 'customer_id',
            ctpty_identifier: str = 'counterparty_id',
            max_onehot_values: int = 100,
    ) -> None:

        self._graph_dataset_definition = graph_dataset_definition
        self._labelled_node_type = graph_dataset_definition.labelled_node_type
        self._max_onehot_values = max_onehot_values

        self._label_col = label_col
        self._timestamp_col = timestamp_col
        self._cust_identifier = cust_identifier
        self._ctpty_identifier = ctpty_identifier

        self._datetime_col = 'DATETIME'
        self._features_attribute_name = 'x'
        self._full_labels_attribute_name = 'LABEL_FULL'
        self._dataset_type_during_hyperparam_opt_attribute_name = 'HYPERPARAM_OPT_DATASET_TYPE'
        self._dataset_type_during_full_training_attribute_name = 'FULL_TRAINING_DATASET_TYPE'
        self._sampler_time_attribute_name = 'time_attr'

        self._train_test_split: TrainTestSplit = self._extract_train_test_split(train_test_split_settings)
        self._node_validity_expressions = self._extract_node_validity_expressions()
        self._graph_data = self._build_graph_data(input_data_path, dataset_filter_exprs, inference_filter_exprs)

    @property
    def graph_data(self) -> HeteroData:
        return self._graph_data

    @property
    def train_test_split(self) -> dict[str, tuple[date, date]]:
        return {
            'TRAIN_HYPERPARAM_OPT': (self._train_test_split.from_date, self._train_test_split.to_date_train_during_hyperparam_opt),
            'TRAIN_FULL': (self._train_test_split.from_date, self._train_test_split.to_date_train_during_full),
            'VALIDATION': (self._train_test_split.from_date_validation, self._train_test_split.to_date_validation),
            'TEST': (self._train_test_split.from_date_test, self._train_test_split.to_date_test),
        }

    @staticmethod
    def _extract_train_test_split(train_test_split_settings: TrainTestSplitSettings) -> TrainTestSplit:
        safe_gap = pl.duration(days=train_test_split_settings.label_safety_gap_days)
        eval_set_len = pl.duration(days=train_test_split_settings.eval_set_len_days)

        from_date = train_test_split_settings.from_date
        to_date_test = train_test_split_settings.to_date - safe_gap
        from_date_test = to_date_test - eval_set_len

        to_date_validation = from_date_test
        from_date_validation = to_date_validation - eval_set_len

        to_date_train_during_full = from_date_test - safe_gap
        to_date_train_during_hyperparam_opt = from_date_validation - safe_gap

        return TrainTestSplit(
            from_date=pl.select(from_date).item(),

            to_date_train_during_hyperparam_opt=pl.select(to_date_train_during_hyperparam_opt).item(),
            to_date_train_during_full=pl.select(to_date_train_during_full).item(),

            from_date_validation=pl.select(from_date_validation).item(),
            to_date_validation=pl.select(to_date_validation).item(),

            from_date_test=pl.select(from_date_test).item(),
            to_date_test=pl.select(to_date_test).item(),
        )

    def _extract_node_validity_expressions(self) -> dict[NodeType, list[pl.Expr]]:
        node_validity_expressions_mapping = {}

        for ntype, node_definition in self._graph_dataset_definition.node_definitions.items():
            valid_node_exprs = [pl.lit(True)]
            node_validity_conditions = node_definition.node_validity_conditions

            if node_validity_conditions.all_not_null is True:
                valid_node_exprs.append(pl.col(node_definition.defining_features).is_not_null())

            if node_validity_conditions.max_n_txs is not None:
                valid_node_exprs.append(pl.len().over(pl.struct(node_definition.defining_features)) < node_validity_conditions.max_n_txs)

            if node_validity_conditions.min_n_unique_cids is not None or node_validity_conditions.max_n_unique_cids is not None:
                min_cids = node_validity_conditions.min_n_unique_cids or 0
                max_cids = node_validity_conditions.max_n_unique_cids or float('inf')
                valid_node_exprs.append(pl.col(self._cust_identifier).n_unique().over(pl.struct(node_definition.defining_features)).is_between(min_cids, max_cids))

            if node_validity_conditions.min_length is not None:
                for min_length, defining_feature in zip(node_validity_conditions.min_length, node_definition.defining_features):
                    if min_length is not None:
                        valid_node_exprs.append(pl.col(defining_feature).str.len_chars() > min_length)

            node_validity_expressions_mapping[ntype] = valid_node_exprs

        return node_validity_expressions_mapping

    @staticmethod
    def _node_id_col_name(ntype: NodeType) -> str:
        return ntype + '_node_id'

    def _build_graph_data(
            self,
            input_data_path: str,
            dataset_filter_exprs: Optional[list[IntoExprColumn]] = None,
            inference_filter_exprs: Optional[list[IntoExprColumn]] = None,
    ) -> HeteroData:

        if dataset_filter_exprs is None:
            dataset_filter_exprs = [pl.lit(True)]
        if inference_filter_exprs is None:
            inference_filter_exprs = [pl.lit(True)]

        columns_to_keep = (
            list(self._graph_dataset_definition.required_columns) +
            [self._timestamp_col, self._label_col] +
            [self._dataset_type_during_hyperparam_opt_attribute_name, self._dataset_type_during_full_training_attribute_name]
        )

        df = (
            pl.scan_parquet(input_data_path)
            .with_columns(pl.from_epoch(self._timestamp_col).alias(self._datetime_col))
            .filter(
                (pl.col(self._datetime_col).is_between(self._train_test_split.from_date, self._train_test_split.to_date_test)),
                *dataset_filter_exprs,
            )
            .with_columns(
                pl.lit(1).alias('ONES'),

                pl.when(pl.col(self._datetime_col).is_between(self._train_test_split.from_date, self._train_test_split.to_date_train_during_hyperparam_opt))
                .then(pl.lit(0))
                .when(pl.col(self._datetime_col).is_between(self._train_test_split.from_date_validation, self._train_test_split.to_date_validation))
                .then(pl.lit(1))
                .when(pl.col(self._datetime_col).is_between(self._train_test_split.from_date_test, self._train_test_split.to_date_test))
                .then(pl.lit(2))
                .otherwise(pl.lit(3))
                .alias(self._dataset_type_during_hyperparam_opt_attribute_name),

                pl.when(pl.col(self._datetime_col).is_between(self._train_test_split.from_date, self._train_test_split.to_date_train_during_full))
                .then(pl.lit(0))
                .when(pl.col(self._datetime_col).is_between(self._train_test_split.from_date_test, self._train_test_split.to_date_test))
                .then(pl.lit(2))
                .otherwise(pl.lit(3))
                .alias(self._dataset_type_during_full_training_attribute_name),
            )
            .select(columns_to_keep)
            .sort(self._timestamp_col, descending=False)
        )

        nodes: dict[NodeType, dict[str, Any]] = {}
        for ntype, node_definition in self._graph_dataset_definition.node_definitions.items():
            if isinstance(node_definition, PrimaryNodeDefinition):
                nodes[ntype] = {}
            node_id_col = self._node_id_col_name(ntype)

            df = (
                df
                .with_columns(pl.all_horizontal(*self._node_validity_expressions[ntype]).alias('valid_node'))
                .with_columns(
                    pl.when(pl.col('valid_node'))
                    .then(pl.struct(node_definition.defining_features).rank(method='dense').sub(1).over('valid_node'))
                    .alias(node_id_col)
                )
                .drop('valid_node')
            )

            if isinstance(node_definition, PrimaryNodeDefinition):

                num_nodes = df.select(node_id_col).max().collect().item() + 1
                nodes[ntype]['num_nodes'] = num_nodes

                if ntype == self._labelled_node_type:
                    sampler_time_df = df.lazy().drop_nulls(node_id_col).group_by(node_id_col).agg(pl.col(self._timestamp_col).max()).collect().to_torch().long()
                else:
                    sampler_time_df = df.lazy().drop_nulls(node_id_col).group_by(node_id_col).agg(pl.col(self._timestamp_col).min()).collect().to_torch().long()

                nodes[ntype][self._sampler_time_attribute_name] = torch.zeros(num_nodes).long()
                nodes[ntype][self._sampler_time_attribute_name][sampler_time_df[:, 0]] = sampler_time_df[:, 1]

                key_columns = [node_id_col, self._label_col, self._dataset_type_during_hyperparam_opt_attribute_name, self._dataset_type_during_full_training_attribute_name]
                associated_feature_names = [feature.source_feature for feature in node_definition.associated_features]
                associated_features = (
                    df
                    .lazy()
                    .drop_nulls(node_id_col)
                    .select(key_columns + associated_feature_names)
                    .pipe(self._preprocess_associated_features, node_definition)
                    .unique(node_id_col, keep='last')
                    .collect()
                    .to_torch(dtype=pl.Float32)
                )

                node_indices = associated_features[:, 0].long()
                nodes[ntype][self._features_attribute_name] = torch.zeros(num_nodes, associated_features.shape[1] - len(key_columns))
                nodes[ntype][self._features_attribute_name][node_indices] = associated_features[:, len(key_columns):]

                if ntype == self._labelled_node_type:

                    label_df = df.lazy().drop_nulls(node_id_col).group_by(node_id_col).agg(
                        pl.col(self._dataset_type_during_hyperparam_opt_attribute_name).min(),
                        pl.col(self._dataset_type_during_full_training_attribute_name).min(),
                        self._label_col.fill_null(False).max(),
                    ).collect().to_torch(dtype=pl.Int64)

                    node_indices = label_df[:, 0]
                    nodes[ntype][self._dataset_type_during_hyperparam_opt_attribute_name] = torch.zeros(num_nodes, dtype=torch.int64)
                    nodes[ntype][self._dataset_type_during_hyperparam_opt_attribute_name][node_indices] = label_df[:, 1]
                    nodes[ntype][self._dataset_type_during_full_training_attribute_name] = torch.zeros(num_nodes, dtype=torch.int64)
                    nodes[ntype][self._dataset_type_during_full_training_attribute_name][node_indices] = label_df[:, 2]
                    nodes[ntype][self._full_labels_attribute_name] = torch.zeros(num_nodes, dtype=torch.int64)
                    nodes[ntype][self._full_labels_attribute_name][node_indices] = label_df[:, 3]

        edges = self._build_edges(df)
        return HeteroData(nodes | edges)

    def _preprocess_associated_features(self, df: pl.LazyFrame | pl.DataFrame, node_defintion: PrimaryNodeDefinition) -> pl.LazyFrame:
        preprocessed_feature_names: list[str] = []
        onehot_feature_names: list[str] = []

        for feature in node_defintion.associated_features:
            if feature.preprocessing_steps:
                preprocessed_feature_names.append(feature.source_feature)
                if feature.preprocessing_steps == ['onehot']:
                    onehot_feature_names.append(feature.source_feature)

        return (
            df.lazy()
            .with_columns(
                pl.col(feature.source_feature).pipe(self._transform_and_fill_associated_feature, feature)
                for feature in node_defintion.associated_features
            )
            .with_columns(
                pl.col(feature).pipe(self._aggregate_rare_categorical_values, feature)
                for feature in onehot_feature_names
            )
            .collect()
            .to_dummies(onehot_feature_names, drop_first=True)
            .lazy()
            .drop(preprocessed_feature_names, strict=False)
        )

    def _transform_and_fill_associated_feature(self, expr: pl.Expr, associated_feature: AssociatedFeatureDefinition) -> pl.Expr:
        match associated_feature.feature_type:
            case 'numeric':
                if associated_feature.preprocessing_steps:
                    expr = self._preprocess_numeric_associated_feature(expr, associated_feature.source_feature, associated_feature.preprocessing_steps)
                expr = expr.fill_null(associated_feature.fill_value).fill_nan(cast(float, associated_feature.fill_value))
                expr = expr.cast(pl.Float64)
            case 'categorical':
                expr = expr.fill_null(associated_feature.fill_value)
                if associated_feature.preprocessing_steps:
                    if associated_feature.preprocessing_steps == ['onehot']:
                        pass
                    else:
                        expr = self._preprocess_categorical_associated_feature(expr, associated_feature.source_feature, associated_feature.preprocessing_steps)
            case 'text':
                # TODO: Do not forget about preprocessing alias to not drop it
                raise ValueError('Preprocessing of text features not supported for now.')
            case _:
                raise ValueError(f'Invalid associated feature type: {associated_feature.feature_type}!')
        return expr

    def _preprocess_numeric_associated_feature(self, expr: pl.Expr, source_feature: str, preprocessing_steps: list[PREPROCESSING_OPTIONS]) -> pl.Expr:
        for preprocessing_step in preprocessing_steps:
            match preprocessing_step:
                case 'log':
                    expr = expr.sign() * expr.abs().add(1).log()
                case 'standardize':
                    mean = expr.filter(pl.col(self._dataset_type_during_full_training_attribute_name) == 0).drop_nulls().drop_nans().mean()
                    std = expr.filter(pl.col(self._dataset_type_during_full_training_attribute_name) == 0).drop_nulls().drop_nans().std()
                    expr = (expr - mean) / (std + 1e-12)
                case _:
                    raise ValueError(f'Invalid preprocessing step for numeric feature type: {preprocessing_step}!')
        expr = expr.alias(source_feature + '_' + '_'.join(preprocessing_steps))
        return expr

    def _preprocess_categorical_associated_feature(self, expr: pl.Expr, source_feature: str, preprocessing_steps: list[PREPROCESSING_OPTIONS]) -> pl.Expr:
        assert len(preprocessing_steps) == 1, f'Categorical features are expected to have only one preprocessing step, received {preprocessing_steps}.'

        preprocessing_step = preprocessing_steps[0]
        match preprocessing_step:
            case 'label':
                expr = expr.rank(method='dense')
            case 'label_norm':
                expr = expr.rank(method='dense')
                expr = expr / expr.max()
            case 'target':
                train_label_mean = self._label_col.filter(pl.col(self._dataset_type_during_full_training_attribute_name) == 0).drop_nulls().drop_nans().mean()
                expr = (
                    pl.when(pl.col(self._dataset_type_during_full_training_attribute_name) == 0)
                    .then(self._label_col)
                    .cumulative_eval(pl.element().mean())
                    .shift(1)
                    .forward_fill()
                    .over(expr)
                    .fill_null(train_label_mean)
                    .fill_null(0.05)
                )
                # expr = pl.int_range(1, pl.len() + 1).over(expr)
            case _:
                raise ValueError(f'Invalid preprocessing step for categoric feature type: {preprocessing_step}!')

        expr = expr.cast(pl.Float64).alias(source_feature + '_' + preprocessing_step)
        return expr

    def _aggregate_rare_categorical_values(self, expr: pl.Expr, feature_name: str) -> pl.Expr:
        common_values = expr.value_counts(sort=True).head(self._max_onehot_values).struct.field(feature_name)
        return pl.when(expr.is_in(common_values)).then(expr).otherwise(pl.lit('RARE_VALUE'))

    def _build_edges(self, preprocessed_dataframe: pl.LazyFrame) -> dict[tuple[NodeType, EdgeDescription, NodeType], dict[str, torch.Tensor]]:

        edges = {}
        for etype in self._graph_dataset_definition.edge_definitions:
            if isinstance(etype, MonoPartiteEdgeType):
                primary_ntype, auxiliary_ntype, edge_desc = etype
                primary_ntype_col = self._node_id_col_name(primary_ntype)
                primary_ntype_col_right = primary_ntype_col + '_right'
                auxiliary_ntype_col = self._node_id_col_name(auxiliary_ntype)
                edgelist = (
                    preprocessed_dataframe
                    .join(preprocessed_dataframe, on=auxiliary_ntype_col, nulls_equal=False)
                    .select(primary_ntype_col, primary_ntype_col_right)
                    .drop_nulls([primary_ntype_col, primary_ntype_col_right])
                    .filter(pl.col(primary_ntype_col) != pl.col(primary_ntype_col_right))
                    .collect()
                    .to_torch()
                    .T.long()
                    .contiguous()
                )

                if (primary_ntype, edge_desc, primary_ntype) not in edges:
                    edges[(primary_ntype, edge_desc, primary_ntype)] = {'edge_index': edgelist}
                else:
                    edges[(primary_ntype, edge_desc, primary_ntype)]['edge_index'] = torch.cat([edges[(primary_ntype, edge_desc, primary_ntype)]['edge_index'], edgelist], dim=1)

            elif isinstance(etype, BiPartiteEdgeType):
                edge_src, edge_desc, edge_dest, reverse_edge_desc = etype
                edgelist = (
                    preprocessed_dataframe.select(pl.col(self._node_id_col_name(edge_src)).alias('edge_src'), pl.col(self._node_id_col_name(edge_dest)).alias('edge_dst'))
                    .drop_nulls()
                    .unique(['edge_src', 'edge_dst'])
                    .collect()
                    .to_torch(dtype=pl.Int64)
                    .T.contiguous()
                )
                edges[(edge_src, edge_desc, edge_dest)] = {'edge_index': edgelist[:2, :]}

                if reverse_edge_desc is not None:
                    edges[(edge_dest, reverse_edge_desc, edge_src)] = {'edge_index': edgelist[[1, 0], :]}

        return edges
