import math
from enum import IntEnum
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
from .. import dependencies, ir, scheduler

from .utils import NodeType, compute_bucket_users, get_node_type
from .bucket_helpers import merge_allgather, merge_ag_wait, merge_reducescatter, merge_rs_wait

def bucket_all_gather_by_block(
    sched: "scheduler.Scheduler",
    snodes: List["scheduler.BaseSchedulerNode"],
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Bucket ALL_GATHER and AG_WAIT by block
    """
    inverse_users, node_users = compute_bucket_users(snodes)

    # get the block each node belongs to
    node_block_list = []
    last_module = ""
    for node in snodes:
        if isinstance(node, scheduler.FusedSchedulerNode):
            node_module = get_block_level(node.snodes[0])
        else:
            node_module = get_block_level(node)
        if node_module == "":
            node_module = last_module
        node_block_list.append(node_module)
        last_module = node_module
    node_block_list = merge_block_name(node_block_list)

    # bucket ALL_GATHER and AG_WAIT by block
    result_list = []
    all_gather_list = []
    all_gather_dep_list = []
    ag_wait_list = []
    node_block_list.reverse()
    last_module = node_block_list[0]

    for idx, node in enumerate(reversed(snodes)):
        current_module = node_block_list[idx]
        if current_module != last_module and len(all_gather_list) > 0:
            # bucketing in the block boundary
            assert len(all_gather_list) == len(ag_wait_list)
            merged_all_gather = merge_allgather(sched, all_gather_list, all_gather_dep_list)
            merged_wait = merge_ag_wait(sched, ag_wait_list, all_gather_list, merged_all_gather[0].node)
    
            for n in merged_wait + merged_all_gather:
                if n not in result_list:
                    result_list.append(n)
            
            ag_wait_list = []
            all_gather_list = []
            all_gather_dep_list = []
                    
        if get_node_type(node) == NodeType.ALL_GATHER:
            # add the small all_gather to bucket
            all_gather_list.append(node)
            all_gather_dep = list(inverse_users[node])
            if len(all_gather_dep) > 0:
                all_gather_dep_list.extend(all_gather_dep)
        elif get_node_type(node) == NodeType.AG_WAIT:
            # add the small ag_wait to bucket
            ag_wait_list.append(node)
        else:
            # we do not bucket other nodes
            if node not in result_list and node not in all_gather_dep_list:
                result_list.append(node)

        last_module = current_module

    assert len(all_gather_list) == len(ag_wait_list)

    if len(all_gather_list) > 0:
        merged_all_gather = merge_allgather(sched, all_gather_list, all_gather_dep_list)
        merged_wait = merge_ag_wait(sched, ag_wait_list, all_gather_list, merged_all_gather[0].node)
        for n in merged_wait + merged_all_gather:
            if n not in result_list:
                result_list.append(n)
        
    result_list.reverse()
 
    return result_list


def bucket_reduce_scatter_by_block(
    sched: "scheduler.Scheduler",
    snodes: List["scheduler.BaseSchedulerNode"],
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Bucket REDUCE_SCATTER and RS_WAIT by block
    """
    inverse_users, node_users = compute_bucket_users(snodes)

    # get the block each node belongs to
    node_block_list = []
    last_module = ""
    for node in snodes:
        if isinstance(node, scheduler.FusedSchedulerNode):
            node_module = get_block_level(node.snodes[0])
        else:
            node_module = get_block_level(node)
        if node_module == "":
            node_module = last_module
        node_block_list.append(node_module)
        last_module = node_module
    node_block_list = merge_block_name(node_block_list)

    # bucket REDUCE_SCATTER and RS_WAIT by block
    result_list = []
    reduce_scatter_list = []
    reduce_scatter_dep_list = []
    rs_wait_list = []
    rs_wait_dep_list = []
    fused_list = []
    last_module = node_block_list[0]
    
    for idx, node in enumerate(snodes):
        current_module = node_block_list[idx]
        if current_module != last_module and len(reduce_scatter_list) > 0:
            # bucketing in the block boundary
            assert len(reduce_scatter_list) == len(rs_wait_list)
            (merged_reduce_scatter, copy_in_size) = merge_reducescatter(sched, reduce_scatter_list, reduce_scatter_dep_list)
            merged_wait = merge_rs_wait(sched, rs_wait_list, rs_wait_dep_list, merged_reduce_scatter[-1].node, copy_in_size)    
            
            for n in merged_reduce_scatter + merged_wait:
                if n not in result_list:
                    result_list.append(n)
                    
            reduce_scatter_list = []
            reduce_scatter_dep_list = []
            rs_wait_list = []
            rs_wait_dep_list = []

        if get_node_type(node) == NodeType.REDUCE_SCATTER:
            # add the small reduce_scatter to bucket
            reduce_scatter_list.append(node)
            inverse_user = list(inverse_users[node])
            if len(inverse_user) > 0:
                if isinstance(inverse_user[0], scheduler.FusedSchedulerNode):
                    fused_list.append(inverse_user[0])
                    for i in inverse_user[0].snodes:
                        if i not in reduce_scatter_dep_list:
                            reduce_scatter_dep_list.append(i)
                            break
                else:
                    reduce_scatter_dep_list.extend(inverse_user)
        elif get_node_type(node) == NodeType.RS_WAIT and "reduce_scatter_tensor" in node.node.inputs[0].python_kernel_name:
            # add the small rs_wait to bucket
            rs_wait_list.append(node)
            rs_wait_dep_list.extend(list(node_users[node]))
        else:
            # we do not bucket other nodes
            if node not in result_list and node not in reduce_scatter_dep_list and node not in rs_wait_dep_list:
                result_list.append(node)
        last_module = current_module
    assert len(reduce_scatter_list) == len(rs_wait_list)
    
    if len(reduce_scatter_list) > 0:
        (merged_reduce_scatter, copy_in_size) = merge_reducescatter(sched, reduce_scatter_list, reduce_scatter_dep_list)
        merged_wait = merge_rs_wait(sched, rs_wait_list, rs_wait_dep_list, merged_reduce_scatter[-1].node, copy_in_size)
        for n in merged_reduce_scatter + merged_wait:
            if n not in result_list:
                result_list.append(n)

    result_list = [r for r in result_list if r not in fused_list]

    return result_list


def merge_block_name(
    node_block_list: List[str],
) -> List[str]:
    """
    Update nodes' block name annotations
    1. Merge the last two blocks as a bigger block, for better overlapping between FWD & BWD
    2. Fix the outlier node block annotation
    """
    # TODO(ruisizhang123): this is an adhoc fix bc compiler module trace bugs
    next_to_last, last = "L['self'].norm", "L['self'].output" 
    first_block = node_block_list[0]
    for i in range(1, len(node_block_list) - 2):
        if (
            node_block_list[i] != node_block_list[i - 1]
            and node_block_list[i] != node_block_list[i + 1]
            and node_block_list[i - 1] == node_block_list[i + 1]
        ):
            # fix outlier node block annotation
            node_block_list[i] = node_block_list[i - 1]
        if first_block == "":
            first_block = node_block_list[i]
    
    for i in range(len(node_block_list)):
        if node_block_list[i] == "":
            node_block_list[i] = first_block
        if node_block_list[i] == next_to_last:
            # merge the last two blocks as a bigger block
            node_block_list[i] = last
    
    for i in range(1, len(node_block_list) - 2):
        if (
            node_block_list[i] != node_block_list[i - 1]
            and node_block_list[i] != node_block_list[i + 1]
            and node_block_list[i - 1] == node_block_list[i + 1]
        ):
            # fix outlier node block annotation
            node_block_list[i] = node_block_list[i - 1]
        if first_block == "":
            first_block = node_block_list[i]

    # make sure every node is annotated
    assert "" not in node_block_list
    return node_block_list

def get_block_level(
    node: "scheduler.BaseSchedulerNode"
) -> str:
    """
    Get the node's block name
    """
    node_origin_list = []
    node_origin_list += node.node.origins
    module_list = []
    for n in node_origin_list:
        module_stack = n.meta.get("nn_module_stack", {})
        if module_stack != {}:
            layer_info, block_info = list(module_stack.values())[0]
            module_list.append(layer_info)
    node_module = list(set(module_list))
    assert len(node_module) == 1 or len(node_module) == 0
    if len(node_module) > 0:
        return node_module[0]
    return ""
