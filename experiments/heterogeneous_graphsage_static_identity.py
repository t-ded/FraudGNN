import json
from pathlib import Path

from src.data_loading.graph_builder import GraphDatasetDefinition
from src.data_loading.tabular_dataset import TabularDatasetDefinition, TrainValTestRatios
from src.evaluation.evaluator import Evaluator, GNNHyperparameters
from src.models.heterogeneous_graphsage import HeteroGraphSAGE

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / 'config/general_config.json'

with open(CONFIG_PATH) as f:
    config = json.load(f)


if __name__ == '__main__':
    tabular_definition = TabularDatasetDefinition(
        data_path=ROOT_DIR / config['processed_data_path'] / 'full_fraud.csv',
        numeric_columns=['city_pop', 'amt'],
        categorical_columns=['job', 'category', 'dob', 'zip'],
        text_columns=[],
        required_columns=['is_fraud'] + ['tx_id', 'dob', 'zip'] + ['city_pop', 'amt'] + ['job', 'category'] + [],
        train_val_test_ratios=TrainValTestRatios(0.8, 0.1, 0.1),
    )

    graph_definition = GraphDatasetDefinition(
        node_feature_cols={'tx_id': ['amt', 'city_pop', 'category'], 'dob': ['dob'], 'zip': ['zip']},
        node_label_cols={'tx_id': 'is_fraud'},
        edge_definitions={
            ('transaction', 'tx_has_dob', 'dob'): ('tx_id', 'dob'),
            ('dob', 'tx_has_dob_reverse', 'transaction'): ('dob', 'tx_id'),
            ('transaction', 'tx_has_zip', 'zip'): ('tx_id', 'zip'),
            ('zip', 'tx_has_zip_reverse', 'transaction'): ('zip', 'tx_id'),
        },
        unique_cols={'tx_id'},
    )

    model = HeteroGraphSAGE(
        in_feats={'transaction': 3, 'dob': 1, 'zip': 1},
        hidden_feats=16,
        n_layers=2,
        out_feats=1,
        edge_definitions=set(graph_definition.edge_definitions.keys()),
        graphsage_aggregator_type='pool',
    )

    evaluator = Evaluator(
        model=model,
        hyperparameters=GNNHyperparameters(learning_rate=0.01, train_batch_size=2_048, validation_batch_size=64),
        tabular_dataset_definition=tabular_definition,
        graph_dataset_definition=graph_definition,
        preprocess_tabular=True,
        identifier='HeterogeneousGraphSAGE_static_identity_only',
        save_logs=True,
    )

    evaluator.train(20, plot_loss=True)
    validation_results = evaluator.stream_evaluate('validation', compute_metrics=True, plot_pr_curve=True)
    validation_results.print_summary(loss_func=evaluator.criterion)
    test_results = evaluator.stream_evaluate('testing', compute_metrics=True, plot_pr_curve=True)
    test_results.print_summary(loss_func=evaluator.criterion)
