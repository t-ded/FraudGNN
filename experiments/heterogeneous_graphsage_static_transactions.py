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
        categorical_columns=['gender', 'job', 'category'],
        text_columns=[],
        required_columns=['is_fraud'] + ['tx_id', 'acct_num', 'merchant'] + ['city_pop', 'amt'] + ['gender', 'job', 'category'] + [],
        train_val_test_ratios=TrainValTestRatios(0.8, 0.1, 0.1),
    )

    graph_definition = GraphDatasetDefinition(
        node_feature_cols={'tx_id': ['amt'], 'acct_num': ['city_pop', 'gender', 'job'], 'merchant': ['category']},
        node_label_cols={'tx_id': 'is_fraud'},
        edge_definitions={
            ('customer', 'sends', 'transaction'): ('acct_num', 'tx_id'),
            ('transaction', 'sent_by', 'customer'): ('tx_id', 'acct_num'),
            ('transaction', 'sent_to', 'merchant'): ('tx_id', 'merchant'),
            ('merchant', 'received', 'transaction'): ('merchant', 'tx_id'),
        },
        unique_cols={'tx_id'},
    )

    model = HeteroGraphSAGE(
        in_feats={'transaction': 1, 'customer': 3, 'merchant': 1},
        hidden_feats=16,
        n_layers=2,
        out_feats=1,
        edge_definitions=set(graph_definition.edge_definitions.keys()),
        graphsage_aggregator_type='pool',
    )

    evaluator = Evaluator(
        model=model,
        hyperparameters=GNNHyperparameters(learning_rate=0.01, batch_size=64),
        tabular_dataset_definition=tabular_definition,
        graph_dataset_definition=graph_definition,
        preprocess_tabular=True,
        identifier='HeterogeneousGraphSAGE',
        save_logs=True,
    )

    evaluator.train(20, plot_loss=True)
    validation_results = evaluator.stream_evaluate('validation', compute_metrics=True, plot_pr_curve=True)
    validation_results.print_summary(loss_func=evaluator.criterion)
    test_results = evaluator.stream_evaluate('testing', compute_metrics=True, plot_pr_curve=True)
    test_results.print_summary(loss_func=evaluator.criterion)
