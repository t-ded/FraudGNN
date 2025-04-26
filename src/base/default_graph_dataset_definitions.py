from src.base.base import NodeType, AuxiliaryNodeDefinition, NodeValidityConditions, PrimaryNodeDefinition, MonoPartiteEdgeType, EdgeDescription, AssociatedFeatureDefinition, BiPartiteEdgeType
from src.base.graph_dataset_definition import GraphDatasetDefinition


class SparkovGraphDatasetDefinitions:
    TX_NO_FEATURES = GraphDatasetDefinition(
        identifier='TX_NO_FEATURES',
        node_definitions={
            NodeType('TRANSACTION'): PrimaryNodeDefinition(
                defining_features=['trans_num'],
                associated_features=[],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('CUSTOMER'): AuxiliaryNodeDefinition(
                defining_features=['cc_num'],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('LAST_NAME'): AuxiliaryNodeDefinition(
                defining_features=['last'],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('CITY'): AuxiliaryNodeDefinition(
                defining_features=['city'],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('JOB'): AuxiliaryNodeDefinition(
                defining_features=['job'],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('DOB'): AuxiliaryNodeDefinition(
                defining_features=['dob'],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('FIRST_ORDER_DATE'): AuxiliaryNodeDefinition(
                defining_features=['first_order_date'],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('FIRST_ORDER_MERCHANT'): AuxiliaryNodeDefinition(
                defining_features=['first_order_merchant'],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
        },
        edge_definitions=[
            MonoPartiteEdgeType(NodeType('TRANSACTION'), NodeType('CUSTOMER'), EdgeDescription('MADE_BY_SAME_CUSTOMER')),
            MonoPartiteEdgeType(NodeType('TRANSACTION'), NodeType('LAST_NAME'), EdgeDescription('CUSTOMER_HAS_SAME_LAST_NAME')),
            MonoPartiteEdgeType(NodeType('TRANSACTION'), NodeType('CITY'), EdgeDescription('CUSTOMER_HAS_SAME_CITY')),
            MonoPartiteEdgeType(NodeType('TRANSACTION'), NodeType('JOB'), EdgeDescription('CUSTOMER_HAS_SAME_JOB')),
            MonoPartiteEdgeType(NodeType('TRANSACTION'), NodeType('DOB'), EdgeDescription('CUSTOMER_HAS_SAME_DOB')),
            MonoPartiteEdgeType(NodeType('TRANSACTION'), NodeType('FIRST_ORDER_DATE'), EdgeDescription('CUSTOMER_HAS_SAME_FIRST_ORDER_DATE')),
            MonoPartiteEdgeType(NodeType('TRANSACTION'), NodeType('FIRST_ORDER_MERCHANT'), EdgeDescription('CUSTOMER_HAS_SAME_FIRST_ORDER_MERCHANT')),
        ],
        labelled_node_type=NodeType('TRANSACTION'),
    )

    TX_IDENTITY_NODE_CENTRIC = GraphDatasetDefinition(
        identifier='TX_IDENTITY_NODE_CENTRIC',
        node_definitions={
            NodeType('TRANSACTION'): PrimaryNodeDefinition(
                defining_features=['trans_num'],
                associated_features=[
                    AssociatedFeatureDefinition(source_feature='category', feature_type='categorical', fill_value='N/A', preprocessing_steps=['onehot']),
                    AssociatedFeatureDefinition(source_feature='amt', feature_type='numeric', fill_value=0.0, preprocessing_steps=['log', 'standardize']),
                ],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('CUSTOMER'): PrimaryNodeDefinition(
                defining_features=['cc_num'],
                associated_features=[
                    AssociatedFeatureDefinition(source_feature='city_pop', feature_type='numeric', fill_value=0.0, preprocessing_steps=['log', 'standardize']),
                ],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('MERCHANT'): PrimaryNodeDefinition(
                defining_features=['merchant'],
                associated_features=[AssociatedFeatureDefinition(source_feature='ONES', feature_type='numeric', fill_value=0.0, preprocessing_steps=None)],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('LAST_NAME'): PrimaryNodeDefinition(
                defining_features=['last'],
                associated_features=[AssociatedFeatureDefinition(source_feature='ONES', feature_type='numeric', fill_value=0.0, preprocessing_steps=None)],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('CITY'): PrimaryNodeDefinition(
                defining_features=['city'],
                associated_features=[AssociatedFeatureDefinition(source_feature='ONES', feature_type='numeric', fill_value=0.0, preprocessing_steps=None)],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('JOB'): PrimaryNodeDefinition(
                defining_features=['job'],
                associated_features=[AssociatedFeatureDefinition(source_feature='ONES', feature_type='numeric', fill_value=0.0, preprocessing_steps=None)],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('DOB'): PrimaryNodeDefinition(
                defining_features=['dob'],
                associated_features=[AssociatedFeatureDefinition(source_feature='ONES', feature_type='numeric', fill_value=0.0, preprocessing_steps=None)],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('FIRST_ORDER_DATE'): PrimaryNodeDefinition(
                defining_features=['first_order_date'],
                associated_features=[AssociatedFeatureDefinition(source_feature='ONES', feature_type='numeric', fill_value=0.0, preprocessing_steps=None)],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('FIRST_ORDER_MERCHANT'): PrimaryNodeDefinition(
                defining_features=['first_order_merchant'],
                associated_features=[AssociatedFeatureDefinition(source_feature='ONES', feature_type='numeric', fill_value=0.0, preprocessing_steps=None)],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
        },
        edge_definitions=[
            BiPartiteEdgeType(NodeType('TRANSACTION'), EdgeDescription('MADE_BY'), NodeType('CUSTOMER'), EdgeDescription('MADE')),
            BiPartiteEdgeType(NodeType('TRANSACTION'), EdgeDescription('MADE_AT'), NodeType('MERCHANT'), EdgeDescription('RECEIVED')),
            BiPartiteEdgeType(NodeType('CUSTOMER'), EdgeDescription('HAS_SAME_LAST_NAME'), NodeType('LAST_NAME'), EdgeDescription('REV_HAS_SAME_LAST_NAME')),
            BiPartiteEdgeType(NodeType('CUSTOMER'), EdgeDescription('HAS_SAME_CITY'), NodeType('CITY'), EdgeDescription('REV_HAS_SAME_CITY')),
            BiPartiteEdgeType(NodeType('CUSTOMER'), EdgeDescription('HAS_SAME_JOB'), NodeType('JOB'), EdgeDescription('REV_HAS_SAME_JOB')),
            BiPartiteEdgeType(NodeType('CUSTOMER'), EdgeDescription('HAS_SAME_DOB'), NodeType('DOB'), EdgeDescription('REV_HAS_SAME_DOB')),
            BiPartiteEdgeType(NodeType('CUSTOMER'), EdgeDescription('HAS_SAME_FIRST_ORDER_DATE'), NodeType('FIRST_ORDER_DATE'), EdgeDescription('REV_HAS_SAME_FIRST_ORDER_DATE')),
            BiPartiteEdgeType(NodeType('CUSTOMER'), EdgeDescription('HAS_SAME_FIRST_ORDER_MERCHANT'), NodeType('FIRST_ORDER_MERCHANT'), EdgeDescription('REV_HAS_SAME_FIRST_ORDER_MERCHANT')),
        ],
        labelled_node_type=NodeType('TRANSACTION'),
    )

    TX_IDENTITY_EDGE_CENTRIC = GraphDatasetDefinition(
        identifier='TX_IDENTITY_EDGE_CENTRIC',
        node_definitions={
            NodeType('TRANSACTION'): PrimaryNodeDefinition(
                defining_features=['trans_num'],
                associated_features=[
                    AssociatedFeatureDefinition(source_feature='category', feature_type='categorical', fill_value='N/A', preprocessing_steps=['onehot']),
                    AssociatedFeatureDefinition(source_feature='amt', feature_type='numeric', fill_value=0.0, preprocessing_steps=['log', 'standardize']),
                ],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('CUSTOMER'): PrimaryNodeDefinition(
                defining_features=['cc_num'],
                associated_features=[
                    AssociatedFeatureDefinition(source_feature='city_pop', feature_type='numeric', fill_value=0.0, preprocessing_steps=['log', 'standardize']),
                ],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('MERCHANT'): PrimaryNodeDefinition(
                defining_features=['merchant'],
                associated_features=[AssociatedFeatureDefinition(source_feature='ONES', feature_type='numeric', fill_value=0.0, preprocessing_steps=None)],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('LAST_NAME'): AuxiliaryNodeDefinition(
                defining_features=['last'],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('CITY'): AuxiliaryNodeDefinition(
                defining_features=['city'],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('JOB'): AuxiliaryNodeDefinition(
                defining_features=['job'],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('DOB'): AuxiliaryNodeDefinition(
                defining_features=['dob'],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('FIRST_ORDER_DATE'): AuxiliaryNodeDefinition(
                defining_features=['first_order_date'],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
            NodeType('FIRST_ORDER_MERCHANT'): AuxiliaryNodeDefinition(
                defining_features=['first_order_merchant'],
                node_validity_conditions=NodeValidityConditions(all_not_null=True),
            ),
        },
        edge_definitions=[
            BiPartiteEdgeType(NodeType('TRANSACTION'), EdgeDescription('MADE_BY'), NodeType('CUSTOMER'), EdgeDescription('MADE')),
            BiPartiteEdgeType(NodeType('TRANSACTION'), EdgeDescription('MADE_AT'), NodeType('MERCHANT'), EdgeDescription('RECEIVED')),
            MonoPartiteEdgeType(NodeType('CUSTOMER'), NodeType('LAST_NAME'), EdgeDescription('HAS_SAME_LAST_NAME')),
            MonoPartiteEdgeType(NodeType('CUSTOMER'), NodeType('CITY'), EdgeDescription('HAS_SAME_CITY')),
            MonoPartiteEdgeType(NodeType('CUSTOMER'), NodeType('JOB'), EdgeDescription('HAS_SAME_JOB')),
            MonoPartiteEdgeType(NodeType('CUSTOMER'), NodeType('DOB'), EdgeDescription('HAS_SAME_DOB')),
            MonoPartiteEdgeType(NodeType('CUSTOMER'), NodeType('FIRST_ORDER_DATE'), EdgeDescription('HAS_SAME_FIRST_ORDER_DATE')),
            MonoPartiteEdgeType(NodeType('CUSTOMER'), NodeType('FIRST_ORDER_MERCHANT'), EdgeDescription('HAS_SAME_FIRST_ORDER_MERCHANT')),
        ],
        labelled_node_type=NodeType('TRANSACTION'),
    )