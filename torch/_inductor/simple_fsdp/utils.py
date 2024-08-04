import math
from enum import IntEnum
from typing import defaultdict, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from .. import dependencies, ir, scheduler
from ..virtualized import V

class NodeType(IntEnum):
    ALL_GATHER = 0
    COMPUTE = 1
    REDUCE_SCATTER = 2
    AG_WAIT = 3
    RS_WAIT = 4

def compute_bucket_users(
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

def get_node_type(node:"scheduler.BaseSchedulerNode") -> NodeType:
    """
    Determine the NodeType of a node
    """
    if isinstance(node, scheduler.FusedSchedulerNode):
        # Only compute nodes are fused
        return NodeType.COMPUTE

    if isinstance(node.node, ir._WaitKernel):
        # Determine if the wait node is waiting for ALL_GATHER or REDUCE_SCATTER
        if (
            node.node.inputs[0].op_overload
            == torch.ops._c10d_functional.all_gather_into_tensor.default
        ):
            return NodeType.AG_WAIT
        elif (
            node.node.inputs[0].op_overload
            == torch.ops._c10d_functional.reduce_scatter_tensor.default
        ):
            return NodeType.RS_WAIT
    elif isinstance(node.node, ir._CollectiveKernel):
        # Determine if the collective kernel is for ALL_GATHER or REDUCE_SCATTER
        if (
            node.node.op_overload
            == torch.ops._c10d_functional.all_gather_into_tensor.default
        ):
            return NodeType.ALL_GATHER
        elif (
            node.node.op_overload
            == torch.ops._c10d_functional.reduce_scatter_tensor.default
        ):
            return NodeType.REDUCE_SCATTER

    elif isinstance(node.node, ir.FallbackKernel):
        # [Only for bucketing]: The copy-in (all_gather_copy_in/chunk_cat) is associated with the newly created ALL_GATHER or REDUCE_SCATTER
        # The copy-out (split_with_sizes_copy/read_out) is associated with the newly created AG_WAIT or RS_WAIT
        if node.node.op_overload == torch.ops.fsdp.split_with_sizes_copy.default:
            return NodeType.AG_WAIT
        elif node.node.op_overload == torch.ops.fsdp.read_out.default:
            return NodeType.RS_WAIT
        elif node.node.op_overload == torch.ops.fsdp.all_gather_copy_in.default:
            return NodeType.ALL_GATHER
        elif node.node.op_overload == torch.ops.fsdp.chunk_cat.default:
            return NodeType.REDUCE_SCATTER

    elif isinstance(node.node, ir.MultiOutput):
        # [Only for bucketing]: Determine if the MultiOutput is associated with the newly created ALL_GATHER or REDUCE_SCATTER
        if (
            isinstance(node.node.inputs[0], ir.FallbackKernel)
            and node.node.inputs[0].op_overload
            == torch.ops.fsdp.all_gather_copy_in.default
        ):
            return NodeType.ALL_GATHER
        if (
            isinstance(node.node.inputs[0], ir.FallbackKernel)
            and node.node.inputs[0].op_overload == torch.ops.fsdp.chunk_cat.default
        ):
            return NodeType.REDUCE_SCATTER

    return NodeType.COMPUTE
