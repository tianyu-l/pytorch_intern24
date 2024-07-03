# mypy: allow-untyped-defs
# pyre-strict
import typing
from typing import (
    Any,
    Counter,
    DefaultDict,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from . import config, ir, comms, scheduler
from collections import OrderedDict

import torch

from .dependencies import WeakDep
from .utils import is_collective, is_wait, tuple_sorted


    
def reorder_all_gather(
    snodes: List["scheduler.BaseSchedulerNode"],
    all_gather_order: Optional[str] = "before",
) -> List["scheduler.BaseSchedulerNode"]:
    '''
    Reorder All Gather and Wait in the forward/backward pass;
    1. all_gather_order set to before: all_gather_i is reordered before wait_i-1
    2. all_gather_order set to after: all_gather_i is reordered after wait_i-1
    '''
    reorder_list: List["scheduler.BaseSchedulerNode"] = []
    all_gather_list: List["scheduler.BaseSchedulerNode"] = []
    node_type_list: List[int] = []

    inverse_users, node_users = comms.compute_node_users(snodes)

    for node in reversed(snodes):
        node_type_list.append(get_node_type(node))

    for idx, node in enumerate(reversed(snodes)):
        node_type = node_type_list[idx]
        if node_type in [2, 3]:
            if node not in reorder_list and node not in all_gather_list:
                reorder_list.append(node)
        elif node_type == 1:
            if all_gather_order == "after" and node_type_list[idx+1]==0 and len(all_gather_list) > 0:
                reorder_list.extend(all_gather_list)
                all_gather_list = []
            reorder_list.append(node)
            if all_gather_order == "before" and node_type_list[idx+1]==0 and len(all_gather_list) > 0: 
                reorder_list.extend(all_gather_list)
                all_gather_list = []
        elif node_type == 0:
            all_gather_list.append(node)
            inverse_user = list(inverse_users[node])
            if len(inverse_user) > 0:
                all_gather_list.extend(inverse_user)
        else:
            raise ValueError("node type not supported")
    if len(all_gather_list) > 0:
        reorder_list.extend(all_gather_list)
    reorder_list.reverse()
    return reorder_list



def reorder_reduce_scatter(
    snodes: List["scheduler.BaseSchedulerNode"]
) -> List["scheduler.BaseSchedulerNode"]:
    '''
    Reorder Reduce Scatter and Wait in the backward pass
    reorder wait_i_rs before reduce_scatter_i+1 
    '''
    reorder_list: List["scheduler.BaseSchedulerNode"] = []
    wait_list: List["scheduler.BaseSchedulerNode"] = []
    node_type_list: List[int] = []
    
    for node in snodes:
        # ruisi: to get the convert_element_type after rs, can be removed after fixing the mpt bug.
        prev_node_type = node_type_list[-2:]
        node_type_list.append(get_node_type(node, prev_node_type))

    for idx, node in enumerate(snodes):
        node_type = node_type_list[idx]
        if node_type == 1:
            if node_type_list[idx-1] == 3:
                wait_list.append(node)
            else:
                reorder_list.append(node)
        elif node_type in [0, 2]:
            reorder_list.append(node)
        elif node_type == 3:
            if len(wait_list) > 0:
                reorder_list.extend(wait_list)
                wait_list = []
            reorder_list.append(node)
        elif node_type == 4:
            wait_list.append(node)
        else:
            raise ValueError("node type not supported")

    if len(wait_list) > 0:
        reorder_list.extend(wait_list)
    return reorder_list


def get_node_type(node, prev_nodes=None) -> int:
    # node_type: {0: all gather; 1: wait_tensor; 2: computation; 3: reduce scatter; 4: convert_element_type after rs}
    # ruisi: we add [4: convert_element_type] bc of some bugs in mpt conversion. after tinayu fixed this, we can remove it
    node_type = 2 
    if not isinstance(node, scheduler.FusedSchedulerNode):
        node_origins_name = " ".join([str(i.name) for i in list(node.node.origins)])
        if "all_gather_into_tensor" in node_origins_name:
            node_type = 0
        if "wait_tensor" in node_origins_name:
            node_type = 1
        if "reduce_scatter_tensor" in node_origins_name:
            node_type = 3
        if "convert_element_type" in node_origins_name and prev_nodes == [3, 1]:
            node_type = 4
    return node_type
