from collections import defaultdict
from enum import IntEnum
from typing import Dict, List, Set, Tuple

import torch

from .. import ir, scheduler


class NodeType(IntEnum):
    ALL_GATHER = 0
    COMPUTE = 1
    REDUCE_SCATTER = 2
    AG_WAIT = 3
    RS_WAIT = 4


def compute_node_users(
    snodes: List["scheduler.BaseSchedulerNode"],
) -> Tuple[
    Dict["scheduler.BaseSchedulerNode", Set["scheduler.BaseSchedulerNode"]],
    Dict["scheduler.BaseSchedulerNode", Set["scheduler.BaseSchedulerNode"]],
]:
    """
    Compute the inverse users and users of each node
    """
    buf_to_snode: Dict[str, scheduler.BaseSchedulerNode] = {}
    for node in snodes:
        if isinstance(node, scheduler.FusedSchedulerNode):
            for x in node.snodes:
                for buf in x.get_outputs():
                    buf_to_snode[buf.get_name()] = node

        for buf in node.get_outputs():
            buf_to_snode[buf.get_name()] = node

    inverse_users = {}
    keys = list(buf_to_snode.keys())
    for node in snodes:
        dep_list = []
        for dep in node.unmet_dependencies:
            if dep.name in keys:
                dep_list.append(buf_to_snode[dep.name])
        inverse_users.update({node: set(dep_list)})

    node_users: Dict[scheduler.BaseSchedulerNode, Set[scheduler.BaseSchedulerNode]] = (
        defaultdict(set)
    )
    for node, node_inverse_users in inverse_users.items():
        for inverse_user in node_inverse_users:
            node_users[inverse_user].add(node)

    return inverse_users, node_users

def get_buffer_type(buffer: "ir.Operation") -> NodeType:
    """
    Determine the type of a buffer
    """
    if isinstance(buffer, ir._WaitKernel):
        # Determine if the wait node is waiting for ALL_GATHER or REDUCE_SCATTER
        if (
            buffer.inputs[0].op_overload
            == torch.ops._c10d_functional.all_gather_into_tensor.default
        ):
            return NodeType.AG_WAIT
        elif (
            buffer.inputs[0].op_overload
            == torch.ops._c10d_functional.reduce_scatter_tensor.default
        ):
            return NodeType.RS_WAIT
    elif isinstance(buffer, ir._CollectiveKernel):
        # Determine if the collective kernel is for ALL_GATHER or REDUCE_SCATTER
        if (
            buffer.op_overload
            == torch.ops._c10d_functional.all_gather_into_tensor.default
        ):
            return NodeType.ALL_GATHER
        elif (
            buffer.op_overload
            == torch.ops._c10d_functional.reduce_scatter_tensor.default
        ):
            return NodeType.REDUCE_SCATTER

    return NodeType.COMPUTE


def get_node_type(node: "scheduler.BaseSchedulerNode") -> NodeType:
    """
    Determine the NodeType of a node
    """
    if isinstance(node, scheduler.FusedSchedulerNode):
        # Only compute nodes are fused
        return NodeType.COMPUTE

    if isinstance(node, scheduler.GroupedSchedulerNode):
        # [Only for bucketing]: newly created AG and RS are grouped as GroupedSchedulerNode
        child_nodes_type = [get_buffer_type(n.node) for n in node.snodes]

        if child_nodes_type[-1] in [NodeType.ALL_GATHER, NodeType.REDUCE_SCATTER]:
            return child_nodes_type[-1]
        elif child_nodes_type[0] in [NodeType.AG_WAIT, NodeType.RS_WAIT]:
            return child_nodes_type[0]
        else:
            return NodeType.COMPUTE

    return get_buffer_type(node.node)
