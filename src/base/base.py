from dataclasses import dataclass
from functools import cached_property
from typing import NewType, Literal, Optional, NamedTuple

NodeType = NewType('NodeType', str)
EdgeDescription = NewType('EdgeDescription', str)
ASSOCIATED_FEATURE_TYPES = Literal['numeric', 'categorical', 'text']
NUMERIC_PREPROCESSING_OPTIONS = Literal['log', 'standardize']
CATEGORICAL_PREPROCESSING_OPTIONS = Literal['onehot', 'label', 'label_norm', 'target']
PREPROCESSING_OPTIONS = NUMERIC_PREPROCESSING_OPTIONS | CATEGORICAL_PREPROCESSING_OPTIONS


@dataclass(frozen=True, kw_only=True)
class AssociatedFeatureDefinition:
    source_feature: str
    feature_type: ASSOCIATED_FEATURE_TYPES
    fill_value: float | str
    preprocessing_steps: Optional[PREPROCESSING_OPTIONS]

    # TODO: Ensure fill value and preprocessing steps correspond to feature type
    def __post_init__(self) -> None:
        pass


@dataclass(frozen=True, kw_only=True)
class NodeValidityConditions:
    all_not_null: bool
    min_n_unique_cids: Optional[int] = None
    min_length: Optional[list[Optional[int]]] = None


@dataclass(frozen=True, kw_only=True)
class NodeDefinition:
    defining_features: list[str]
    node_validity_conditions: NodeValidityConditions = NodeValidityConditions(all_not_null=False, min_n_unique_cids=None, min_length=None)

    @cached_property
    def required_columns(self) -> set[str]:
        return set(self.defining_features)


@dataclass(frozen=True, kw_only=True)
class PrimaryNodeDefinition(NodeDefinition):
    associated_features: list[AssociatedFeatureDefinition]

    @cached_property
    def required_columns(self) -> set[str]:
        return set(self.defining_features) | {associated_feature_def.source_feature for associated_feature_def in self.associated_features}


@dataclass(frozen=True, kw_only=True)
class AuxiliaryNodeDefinition(NodeDefinition):
    pass


class MonoPartiteEdgeType(NamedTuple):
    primary_ntype: NodeType
    auxiliary_ntype: NodeType | list[NodeType]
    edge_description: EdgeDescription


class BiPartiteEdgeType(NamedTuple):
    source_ntype: NodeType
    edge_description: EdgeDescription
    destination_ntype: NodeType
    reverse_edge_description: Optional[EdgeDescription] = None


@dataclass(kw_only=True)
class GNNInferenceHyperparameters:
    sampler_fanouts: list[int]
    prediction_threshold: float
