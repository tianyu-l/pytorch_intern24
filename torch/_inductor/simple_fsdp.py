# mypy: allow-untyped-defs
# pyre-strict
from collections import defaultdict
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

import torch

from . import ir, scheduler


class NodeType(IntEnum):
    ALL_GATHER = 0
    WAIT = 1
    COMPUTE = 2
    REDUCE_SCATTER = 3


def compute_node_users(
    snodes: List["scheduler.BaseSchedulerNode"],
) -> Tuple[
    Dict["scheduler.BaseSchedulerNode", Set["scheduler.BaseSchedulerNode"]],
    Dict["scheduler.BaseSchedulerNode", Set["scheduler.BaseSchedulerNode"]],
]:
    # set up buffer name to (fused)snode mapping
    buf_to_snode: Dict[str, scheduler.BaseSchedulerNode] = {}
    for node in snodes:
        if isinstance(node, scheduler.FusedSchedulerNode):
            for x in node.snodes:
                for buf in x.get_outputs():
                    buf_to_snode[buf.get_name()] = node

        for buf in node.get_outputs():
            buf_to_snode[buf.get_name()] = node
    # compute inverse_users
    inverse_users = {}
    keys = list(buf_to_snode.keys())
    for node in snodes:
        dep_list = []
        for dep in node.unmet_dependencies:
            if dep.name in keys:
                dep_list.append(buf_to_snode[dep.name])
        inverse_users.update({node: set(dep_list)})

    # compute node_users
    # TODO: ideally, we should deduplicate .users and .node_users,
    # but currently .users contains extra information that's difficult to
    # extract into a standalone container.
    node_users: Dict[scheduler.BaseSchedulerNode, Set[scheduler.BaseSchedulerNode]] = (
        defaultdict(set)
    )
    for node, node_inverse_users in inverse_users.items():
        for inverse_user in node_inverse_users:
            node_users[inverse_user].add(node)

    return inverse_users, node_users


def reorder_all_gather(
    snodes: List["scheduler.BaseSchedulerNode"],
    all_gather_before_last_wait: Optional[bool] = True,
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Reorder All Gather and Wait in the forward/backward pass;
    1. all_gather_before_last_wait set to True: all_gather_i is reordered before wait_i-1
    2. all_gather_before_last_wait set to False: all_gather_i is reordered after wait_i-1
    """
    result_list: List[scheduler.BaseSchedulerNode] = []
    all_gather_list: List[scheduler.BaseSchedulerNode] = []
    node_to_type: Dict[scheduler.BaseSchedulerNode, int] = {}

    inverse_users, node_users = compute_node_users(snodes)
    snodes.reverse()

    for node in snodes:
        node_to_type[node] = get_node_type(node)

    for idx, node in enumerate(snodes):
        node_type = node_to_type[node]
        if node_type in [NodeType.REDUCE_SCATTER, NodeType.COMPUTE]:
            # we do not reorder reduce scatter and compute node
            if node not in result_list and node not in all_gather_list:
                result_list.append(node)
        elif node_type == NodeType.ALL_GATHER:
            # gather i-th all gather node and its dependencies
            all_gather_list.append(node)
            inverse_user = list(inverse_users[node])
            if len(inverse_user) > 0:
                all_gather_list.extend(inverse_user)
        elif node_type == NodeType.WAIT:
            if (
                node_to_type[snodes[idx + 1]] == NodeType.ALL_GATHER
                and not all_gather_before_last_wait
                and len(all_gather_list) > 0
            ):
                # move i-th all gather node and its dependencies after (i-1)-th wait node (bc this is a reverse list)
                result_list.extend(all_gather_list)
                all_gather_list = []
            # add wait node
            result_list.append(node)
            if (
                node_to_type[snodes[idx + 1]] == NodeType.ALL_GATHER
                and all_gather_before_last_wait
                and len(all_gather_list) > 0
            ):
                # move i-th all gather node and its dependencies before (i-1)-th wait node (bc this is a reverse list)
                result_list.extend(all_gather_list)
                all_gather_list = []

    if len(all_gather_list) > 0:
        result_list.extend(all_gather_list)
    result_list.reverse()
    return result_list


def reorder_reduce_scatter(
    snodes: List["scheduler.BaseSchedulerNode"],
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Reorder Reduce Scatter and Wait in the backward pass
    reorder wait_i_rs before reduce_scatter_i+1
    """
    result_list: List[scheduler.BaseSchedulerNode] = []
    wait_list: List[scheduler.BaseSchedulerNode] = []
    node_to_type: Dict[scheduler.BaseSchedulerNode, int] = {}

    inverse_users, node_users = compute_node_users(snodes)

    for node in snodes:
        node_to_type[node] = get_node_type(node)

    for idx, node in enumerate(snodes):
        node_type = node_to_type[node]
        if node_type in [NodeType.ALL_GATHER, NodeType.COMPUTE]:
            # we do not reorder all gather and compute node
            if node not in wait_list:
                result_list.append(node)
        elif node_type == NodeType.WAIT:
            if node_to_type[snodes[idx - 1]] == NodeType.REDUCE_SCATTER:
                # gather wait node after reduce scatter
                wait_list.append(node)
                wait_list.extend(node_users[node])
            else:
                # we do not reorder wait node after all gather
                result_list.append(node)
        elif node_type == NodeType.REDUCE_SCATTER:
            if len(wait_list) > 0:
                # move the i-th wait node before (i+1)-th reduce scatter node
                result_list.extend(wait_list)
                wait_list = []
            # add reduce scatter node
            result_list.append(node)

    if len(wait_list) > 0:
        result_list.extend(wait_list)
    return result_list


def get_node_type(node) -> int:
    if isinstance(node, scheduler.FusedSchedulerNode):
        return NodeType.COMPUTE

    if isinstance(node.node, ir._WaitKernel):
        return NodeType.WAIT
    elif isinstance(node.node, ir._CollectiveKernel):
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

    return NodeType.COMPUTE
