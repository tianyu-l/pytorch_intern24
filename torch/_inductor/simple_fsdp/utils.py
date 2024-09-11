from collections import defaultdict
from enum import IntEnum
from typing import Dict, List, Set, Tuple

import torch

from .. import config, ir, scheduler


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


def _check_ir_node_fsdp(ir_node: "ir.Operation") -> bool:
    """
    Determine if the AG/RS node is for FSDP or TP
    """
    if config.simplefsdp.tp_degree < 0:
        return True

    is_fsdp = False
    ir_node_origins = list(getattr(ir_node, "origins", None))

    if len(ir_node_origins) == 0:
        # bucketed AG and RS doesn't have origins, but they are created by FSDP
        is_fsdp = True

    for n in ir_node_origins:
        meta_data = n.meta.get("stack_trace", {})
        # TODO(ruisizhang123): hack to get FSDP node (the FSDP AG/RS are created from torch_spmd)
        if "parametrization" in meta_data:
            is_fsdp = True
    return is_fsdp


def _get_ir_node_type(ir_node: "ir.Operation") -> NodeType:
    """
    Determine the type of a ir node
    """
    if isinstance(ir_node, ir._WaitKernel):
        # Determine if the wait node is waiting for ALL_GATHER or REDUCE_SCATTER
        ir_op_overload = getattr(ir_node.inputs[0], "op_overload", None)
        if (
            ir_op_overload == torch.ops._c10d_functional.all_gather_into_tensor.default
            and _check_ir_node_fsdp(ir_node.inputs[0])
        ):
            return NodeType.AG_WAIT
        elif (
            ir_op_overload == torch.ops._c10d_functional.reduce_scatter_tensor.default
            and _check_ir_node_fsdp(ir_node.inputs[0])
        ):
            return NodeType.RS_WAIT
    elif isinstance(ir_node, ir._CollectiveKernel):
        # Determine if the collective kernel is for ALL_GATHER or REDUCE_SCATTER
        ir_op_overload = getattr(ir_node, "op_overload", None)
        if (
            ir_op_overload == torch.ops._c10d_functional.all_gather_into_tensor.default
            and _check_ir_node_fsdp(ir_node)
        ):
            return NodeType.ALL_GATHER
        elif (
            ir_op_overload == torch.ops._c10d_functional.reduce_scatter_tensor.default
            and _check_ir_node_fsdp(ir_node)
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
        child_nodes_type = [
            _get_ir_node_type(n) for n in [node.snodes[0].node, node.snodes[-1].node]
        ]
        if child_nodes_type[0] in [NodeType.AG_WAIT, NodeType.RS_WAIT]:
            return child_nodes_type[0]
        elif child_nodes_type[1] in [NodeType.ALL_GATHER, NodeType.REDUCE_SCATTER]:
            return child_nodes_type[1]
        else:
            return NodeType.COMPUTE

    return _get_ir_node_type(node.node)
