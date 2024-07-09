# mypy: allow-untyped-defs
# pyre-strict
from enum import IntEnum
from typing import Dict, List, Optional

import torch
from . import comms, ir, scheduler


class NodeType(IntEnum):
    ALL_GATHER = 0
    WAIT = 1
    COMPUTE = 2
    REDUCE_SCATTER = 3
    CONVERT_ELEMENT_TYPE = 4


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

    inverse_users, node_users = comms.compute_node_users(snodes)
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
        else:
            raise ValueError("node type not supported")
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

    prev_node_type: List[int] = []
    for node in snodes:
        # TODO(ruisizhang123): to get the convert_element_type after rs, can be removed after fixing the mpt bug.
        cur_node_type = get_node_type(node, prev_node_type[-2:])
        node_to_type[node] = cur_node_type
        prev_node_type.append(cur_node_type)

    for idx, node in enumerate(snodes):
        node_type = node_to_type[node]
        if node_type in [NodeType.ALL_GATHER, NodeType.COMPUTE]:
            # we do not reorder all gather and compute node
            result_list.append(node)
        elif node_type == NodeType.WAIT:
            if node_to_type[snodes[idx - 1]] == NodeType.REDUCE_SCATTER:
                # gather wait node after reduce scatter
                wait_list.append(node)
            else:
                # we do not reorder wait node after all gather
                result_list.append(node)
        elif node_type == NodeType.CONVERT_ELEMENT_TYPE:
            # TODO(ruisizhang123): gather reduce scatter wait's follow-up covert element, can be removed after fixing the mpt bug.
            wait_list.append(node)
        elif node_type == NodeType.REDUCE_SCATTER:
            if len(wait_list) > 0:
                # move the i-th wait node before (i+1)-th reduce scatter node
                result_list.extend(wait_list)
                wait_list = []
            # add reduce scatter node
            result_list.append(node)
        else:
            raise ValueError("node type not supported")

    if len(wait_list) > 0:
        result_list.extend(wait_list)
    return result_list


def get_node_type(node, prev_nodes=None) -> int:
    # node_type: {0: all gather; 1: wait_tensor; 2: computation; 3: reduce scatter; 4: convert_element_type after rs}
    node_type = NodeType.COMPUTE
    if not isinstance(node, scheduler.FusedSchedulerNode):
        if (
            isinstance(node.node, ir._CollectiveKernel)
            and node.node.op_overload
            == torch.ops._c10d_functional.all_gather_into_tensor.default
        ):
            node_type = NodeType.ALL_GATHER
        if isinstance(node.node, ir._WaitKernel):
            node_type = NodeType.WAIT
        if (
            isinstance(node.node, ir._CollectiveKernel)
            and node.node.op_overload
            == torch.ops._c10d_functional.reduce_scatter_tensor.default
        ):
            node_type = NodeType.REDUCE_SCATTER
        # TODO(ruisizhang123): we add [4: convert_element_type] bc of some bugs in mpt conversion. after tianyu fixed this, we can remove it
        if isinstance(node.node, ir.ComputedBuffer) and prev_nodes == [
            NodeType.REDUCE_SCATTER,
            NodeType.WAIT,
        ]:
            node_type = NodeType.CONVERT_ELEMENT_TYPE
    return node_type
