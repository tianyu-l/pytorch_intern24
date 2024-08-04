# mypy: allow-untyped-defs
# pyre-strict
import math
from enum import IntEnum
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from .. import dependencies, ir, scheduler
from ..virtualized import V

from .utils import NodeType, compute_bucket_users, get_node_type


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
    inverse_users, node_users = compute_bucket_users(snodes)
    
    for node in snodes:
        node_to_type[node] = get_node_type(node)
    
    snodes.reverse()
    for idx, node in enumerate(snodes):
        node_type = node_to_type[node]
        if node_type in [NodeType.REDUCE_SCATTER, NodeType.COMPUTE, NodeType.RS_WAIT]:
            # we do not reorder reduce scatter and compute node
            if node not in result_list and node not in all_gather_list:
                result_list.append(node)
        elif node_type == NodeType.ALL_GATHER:
            # gather i-th all gather node and its dependencies
            all_gather_list.append(node)
            inverse_user = list(inverse_users[node])
            if (
                len(inverse_user) > 0
                and not node_to_type[inverse_user[0]] == NodeType.ALL_GATHER
                and inverse_user not in all_gather_list
            ):
                all_gather_list.extend(inverse_user)
        elif node_type == NodeType.AG_WAIT:
            if (
                (
                    (
                        node_to_type[snodes[idx + 1]] == NodeType.AG_WAIT
                        and node_to_type[snodes[idx + 2]] == NodeType.ALL_GATHER
                    )
                    or (
                        node_to_type[snodes[idx + 1]] == NodeType.ALL_GATHER
                        and node_to_type[snodes[idx - 1]] != NodeType.AG_WAIT
                    )
                )
                and not all_gather_before_last_wait
                and len(all_gather_list) > 0
            ):
                # move i-th all gather node and its dependencies after (i-1)-th wait node (bc this is a reverse list)
                result_list.extend(all_gather_list)
                all_gather_list = []

            result_list.append(node)

            if (
                (
                    (
                        node_to_type[snodes[idx - 1]] == NodeType.AG_WAIT
                        and node_to_type[snodes[idx + 1]] == NodeType.ALL_GATHER
                    )
                    or (
                        node_to_type[snodes[idx + 1]] == NodeType.ALL_GATHER
                        and node_to_type[snodes[idx - 1]] != NodeType.AG_WAIT
                    )
                )
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
    front_node: "scheduler.BaseSchedulerNode"
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Reorder Reduce Scatter and Wait in the backward pass
    reorder wait_i_rs before reduce_scatter_i+1
    """
    result_list: List[scheduler.BaseSchedulerNode] = []
    wait_list: List[scheduler.BaseSchedulerNode] = []
    node_to_type: Dict[scheduler.BaseSchedulerNode, int] = {}
    inverse_users, node_users = compute_bucket_users(snodes)
    types = []
    for node in snodes:
        node_to_type[node] = get_node_type(node)
        types.append(get_node_type(node))

    for idx, node in enumerate(snodes):
        node_type = node_to_type[node]
        if node_type in [NodeType.ALL_GATHER, NodeType.COMPUTE, NodeType.AG_WAIT]:
            if node not in result_list and node not in wait_list:
                result_list.append(node)
        elif node_type == NodeType.RS_WAIT:
            if node_to_type[snodes[idx - 1]] == NodeType.REDUCE_SCATTER or (
                node_to_type[snodes[idx - 2]] == NodeType.REDUCE_SCATTER
                and node_to_type[snodes[idx - 1]] == NodeType.RS_WAIT
            ):
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

    # minor adjust to ensure overlapping between FWD & BWD
    pick_up_wait = []
    new_result_list = []
    picked = False
    append = False
    for i in result_list:
        if not picked and get_node_type(i) == NodeType.AG_WAIT:
            pick_up_wait.append(i)
            if len(pick_up_wait) == 2:
                picked = True 
        else:
            if i not in pick_up_wait:
                new_result_list.append(i)    
            if not append and picked and i == front_node:
                new_result_list.extend(pick_up_wait)
                append = True  
            if append and picked:
                break
    result_list[:len(new_result_list)]  = new_result_list
     
    return result_list


def get_front_node(
    snodes: List["scheduler.BaseSchedulerNode"],
) -> "scheduler.BaseSchedulerNode":
    """
    Get the front node used for hide BWD communication
    """
    i = 0
    inverse_users, node_users = compute_bucket_users(snodes)
    front_node = snodes[0]
    see_other_type = False
    for node in snodes:
        if get_node_type(node) == NodeType.COMPUTE and not see_other_type:
            users = [get_node_type(i) for i in list(node_users[node])]
            if NodeType.REDUCE_SCATTER in users:
                front_node = node
                break
        else:
            see_other_type = True
            break
    return front_node
