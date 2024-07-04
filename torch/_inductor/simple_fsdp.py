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
from enum import Enum
from . import config, ir, comms, scheduler
from collections import OrderedDict

import torch

from .dependencies import WeakDep
from .utils import is_collective, is_wait, tuple_sorted

import torch.distributed._functional_collectives
_c10d_functional = torch.ops._c10d_functional

class NodeType(Enum):
    ALL_GATHER = 0
    WAIT = 1
    COMP = 2
    REDUCE_SCATTER = 3
    COVERT_ELEMENT = 4

def reorder_all_gather(
    snodes: List["scheduler.BaseSchedulerNode"],
    all_gather_before_last_wait: Optional[bool] = True,
) -> List["scheduler.BaseSchedulerNode"]:
    '''
    Reorder All Gather and Wait in the forward/backward pass;
    1. all_gather_before_last_wait set to True: all_gather_i is reordered before wait_i-1
    2. all_gather_before_last_wait set to False: all_gather_i is reordered after wait_i-1
    '''
    reorder_list: List["scheduler.BaseSchedulerNode"] = []
    all_gather_list: List["scheduler.BaseSchedulerNode"] = []
    node_type_list: List[int] = []

    inverse_users, node_users = comms.compute_node_users(snodes)

    for node in reversed(snodes):
        node_type_list.append(get_node_type(node))

    for idx, node in enumerate(reversed(snodes)):
        node_type = node_type_list[idx]
        if node_type in [NodeType.REDUCE_SCATTER, NodeType.COMP]:
            # we do not reorder reduce scatter and computation node
            if node not in reorder_list and node not in all_gather_list:
                reorder_list.append(node)
        elif node_type == NodeType.ALL_GATHER:
            # gather i-th all gather node and its dependencies
            all_gather_list.append(node)
            inverse_user = list(inverse_users[node])
            if len(inverse_user) > 0:
                all_gather_list.extend(inverse_user)
        elif node_type == NodeType.WAIT:
            if not all_gather_before_last_wait and node_type_list[idx+1]==NodeType.ALL_GATHER and len(all_gather_list) > 0:
                # move i-th all gather node and its dependencies after (i-1)-th wait node (bc this is a reverse list)
                reorder_list.extend(all_gather_list)
                all_gather_list = []
            # add wait node
            reorder_list.append(node)
            if all_gather_before_last_wait and node_type_list[idx+1]==NodeType.ALL_GATHER and len(all_gather_list) > 0: 
                # move i-th all gather node and its dependencies before (i-1)-th wait node (bc this is a reverse list)
                reorder_list.extend(all_gather_list)
                all_gather_list = []
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
        # TODO(ruisizhang123): to get the convert_element_type after rs, can be removed after fixing the mpt bug.
        prev_node_type = node_type_list[-2:]
        node_type_list.append(get_node_type(node, prev_node_type))

    for idx, node in enumerate(snodes):
        node_type = node_type_list[idx]
        if node_type in [NodeType.ALL_GATHER, NodeType.COMP]:
            # we do not reorder all gather and comp node
            reorder_list.append(node)
        elif node_type == NodeType.WAIT:
            if node_type_list[idx-1] == NodeType.REDUCE_SCATTER:
                # gather wait node after reduce scatter
                wait_list.append(node)
            else:
                # we do not reorder wait node after all gather
                reorder_list.append(node)
        elif node_type == NodeType.COVERT_ELEMENT:
            # TODO(ruisizhang123): gather reduce scatter wait's follow-up covert element, can be removed after fixing the mpt bug.
            wait_list.append(node)
        elif node_type == NodeType.REDUCE_SCATTER:
            if len(wait_list) > 0:
                # move the i-th wait node before (i+1)-th reduce scatter node
                reorder_list.extend(wait_list)
                wait_list = []
            # add reduce scatter node
            reorder_list.append(node)
        else:
            raise ValueError("node type not supported")

    if len(wait_list) > 0:
        reorder_list.extend(wait_list)
    return reorder_list


def get_node_type(node, prev_nodes=None) -> int:
    # node_type: {0: all gather; 1: wait_tensor; 2: computation; 3: reduce scatter; 4: convert_element_type after rs}
    # TODO(ruisizhang123): we add [4: convert_element_type] bc of some bugs in mpt conversion. after tianyu fixed this, we can remove it
    node_type = NodeType.COMP
    if not isinstance(node, scheduler.FusedSchedulerNode):
        node_origins_name = " ".join([str(i.name) for i in list(node.node.origins)])
        if "all_gather_into_tensor" in node_origins_name:
            node_type = NodeType.ALL_GATHER
        if "wait_tensor" in node_origins_name:
            node_type = NodeType.WAIT
        if "reduce_scatter_tensor" in node_origins_name:
            node_type = NodeType.REDUCE_SCATTER
        if "convert_element_type" in node_origins_name and prev_nodes == [NodeType.REDUCE_SCATTER, NodeType.WAIT]:
            node_type = NodeType.COVERT_ELEMENT
    return node_type


def bucketing_per_blcok(
    snodes: List["scheduler.BaseSchedulerNode"],
) -> List["scheduler.BaseSchedulerNode"]:
        
    return snodes
