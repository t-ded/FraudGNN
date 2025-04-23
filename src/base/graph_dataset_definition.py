from dataclasses import dataclass
from functools import cached_property

from src.base.base import NodeType, NodeDefinition, MonoPartiteEdgeType, BiPartiteEdgeType, EdgeDescription


@dataclass(frozen=True, kw_only=True)
class GraphDatasetDefinition:
    identifier: str

    node_definitions: dict[NodeType, NodeDefinition]
    edge_definitions: list[MonoPartiteEdgeType | BiPartiteEdgeType]

    labelled_node_type: NodeType

    # TODO: Add validity checks on compatibility of node and edge definitions
    def __post_init__(self) -> None:
        pass

    @cached_property
    def required_columns(self) -> set[str]:
        return set.union(*(node_def.required_columns for node_def in self.node_definitions.values()))

    @cached_property
    def canonical_edges(self) -> list[tuple[NodeType, EdgeDescription, NodeType]]:
        canonical_edges = []

        for edge_def in self.edge_definitions:
            if isinstance(edge_def, MonoPartiteEdgeType):
                primary_ntype, _, edge_desc = edge_def
                canonical_edges.append((primary_ntype, edge_desc, primary_ntype))
            elif isinstance(edge_def, BiPartiteEdgeType):
                src_ntype, edge_desc, dst_ntype, reverse_edge_desc = edge_def
                canonical_edges.append((src_ntype, edge_desc, dst_ntype))
                if reverse_edge_desc is not None:
                    canonical_edges.append((dst_ntype, reverse_edge_desc, src_ntype))

        return canonical_edges