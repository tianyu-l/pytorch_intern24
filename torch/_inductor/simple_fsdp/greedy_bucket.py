from dataclasses import dataclass, field
from typing import Dict, List, Union

import torch
import torch.distributed as dist
from .. import ir, scheduler

from .utils import NodeType, compute_node_users, get_node_type
from .bucket import merge_allgather, merge_ag_wait, merge_reducescatter, merge_rs_wait
from ..comm_analysis import estimate_bucked_nccl_collective_runtime

@dataclass
class AG_INFO:
    AG_INV_DEP: List["scheduler.BaseSchedulerNode"] = field(default_factory=list)
    AG_WAIT: List["scheduler.BaseSchedulerNode"] = field(default_factory=list)
    COMPUTE: List["scheduler.BaseSchedulerNode"] = field(default_factory=list)
    AG_TIME: float = 0
    COMPUTE_TIME: float = 0
    COMPUTE_MEMORY: float = 0

    # additional configs for backward
    REDUCE_SCATTER: List["scheduler.BaseSchedulerNode"] = field(default_factory=list)
    RS_WAIT: List["scheduler.BaseSchedulerNode"] = field(default_factory=list)
    RS_WAIT_DEP: List["scheduler.BaseSchedulerNode"] = field(default_factory=list)
    RS_TIME: float = 0


def greedy_check(current_comp, current_ag, current_mem, node_ag, node_mem, memory_constraint, first_AG, current_rs=0):
    """
    Greedy algorithm for backward
    """
    if current_ag == 0:
        return False

    if first_AG:
        return True

    if current_mem + node_mem > memory_constraint:
        return True

    if current_ag + node_ag + current_rs > current_comp:
        return True

    return False

def get_ag_info(
    snodes: List["scheduler.BaseSchedulerNode"],
    run_time_dict: Dict[str, List[Union[str, float, float]]],
    stage="forward"
) -> Dict["scheduler.BaseSchedulerNode", AG_INFO]:
    """
    Get the information of all_gather and reduce_scatter
    """
    assert stage in ("forward", "backward")

    inverse_users, node_users = compute_node_users(snodes)
    front_nodes = []
    all_gather = None
    ag_info_dict = {}

    # pick up the computes that don't have corresponding all_gather
    for node in snodes:
        users_type = [get_node_type(i) for i in list(node_users[node])]
        if NodeType.ALL_GATHER in users_type:
            break
        front_nodes.append(node)

    for node in snodes[len(front_nodes):]:
        if node.get_name() in run_time_dict:
            _, run_time, memory = run_time_dict[node.get_name()]
        else:
            run_time, memory = 0, 0

        if get_node_type(node) == NodeType.ALL_GATHER:
            # A ag_info consists of 9 parts:
            # [Node]: (1) AG_INV_DEP: The node ALL_GATHER depends on for read-in; (2) AG_WAIT: ALL_GATHER's wait nodes; (3) COMPUTE: COMPUTE nodes ALL_GATHER fetches.
            # （4） REDUCE_SCATTER: REDUCE_SCATTE that reads from COMPUTE nodes; (5) RS_WAIT: REDUCE_SCATTER's wait nodes;
            # [Estimation Num.]: (1) AG_TIME: The estimated ALL_GATHER time; (2) COMPUTE_TIME: The estimated COMPUTE time; (3) COMPUTE_MEMORY: The estimated meory for computation.
            # (4) RS_TIME: The estimated REDUCE_SCATTER time; 
            ag_info = AG_INFO()
            ag_wait = list(node_users[node])

            ag_info.AG_WAIT = ag_wait
            ag_info.AG_TIME = run_time
            ag_info_dict[node] = ag_info
            all_gather = node

            if len(front_nodes) > 0:
                ag_info_dict[all_gather].AG_INV_DEP.extend(front_nodes)
                front_nodes = []
            ag_info_dict[all_gather].AG_INV_DEP.extend(list(inverse_users[node]))
            
        if get_node_type(node) == NodeType.COMPUTE:
            # if the compute node is the inverse user of AG and user of RS_Wait, we should group them with the next AG or last RS_Wait
            users_type = [get_node_type(i) for i in list(node_users[node])]
            if NodeType.ALL_GATHER in users_type:
                continue
            if stage == "backward":
                inverse_users_type = [get_node_type(i) for i in list(inverse_users[node])]
                if NodeType.RS_WAIT in inverse_users_type:
                    continue
            ag_info_dict[all_gather].COMPUTE.append(node)
            ag_info_dict[all_gather].COMPUTE_TIME += run_time
            ag_info_dict[all_gather].COMPUTE_MEMORY += memory

        if stage == "backward":
            if get_node_type(node) == NodeType.REDUCE_SCATTER:
                # rs_dep --> rs --> rs_wait --> rs_wait_dep
                ag_info_dict[all_gather].REDUCE_SCATTER.append(node)
                ag_info_dict[all_gather].RS_TIME += run_time
                    
            if get_node_type(node) == NodeType.RS_WAIT:
                ag_info_dict[all_gather].RS_WAIT.append(node)
                ag_info_dict[all_gather].RS_WAIT_DEP.extend(list(node_users[node]))

    # make sure all nodes are indexed in ag_info_dict
    count = 0
    for key, value in ag_info_dict.items():
        count += len(value.AG_INV_DEP) + len(value.AG_WAIT) + len(value.COMPUTE) + len(value.REDUCE_SCATTER) + len(value.RS_WAIT) + len(value.RS_WAIT_DEP)
        count += 1 
    assert count == len(snodes)
    
    return ag_info_dict


def get_greedy_bucket_plan(
    sched: "scheduler.Scheduler",
    snodes: List["scheduler.BaseSchedulerNode"],
    ag_info_dict: Dict["scheduler.BaseSchedulerNode", AG_INFO],
    stage="forward"
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Greedy Bucket ALL_GATHER and REDUCE_SCATTER
    """
    assert stage in ("forward", "backward")
    
    result_list = []
    all_gather_list = []
    all_gather_dep_list = []
    ag_wait_list = []
    compute_list = []

    current_comp = 0 # compute time in step_i
    current_ag = 0 # all gather time in step_i
    current_mem = 0 # memory in step_i
    current_rs = 0 # reduce scatter time in step_i, by default, it is 0 in forward
    next_comp = 0 # compute time in step_(i+1), derived from all gather in step_i
    first_AG = True

    if stage == "backward":
        reduce_scatter_list = []
        rs_wait_list = []
        rs_wait_dep_list = []
        next_rs = 0 # reduce scatter time in step_(i+1), derived from all gather in step_i

    for idx, node in enumerate(snodes):
        if get_node_type(node) == NodeType.ALL_GATHER: 
            if greedy_check(current_comp, current_ag, current_mem, ag_info_dict[node].AG_TIME, ag_info_dict[node].COMPUTE_MEMORY, 320, first_AG, current_rs):
                # merge all_gather  
                merged_all_gather, ag_buffer = merge_allgather(sched, all_gather_list)
                merged_ag_wait = merge_ag_wait(sched, ag_wait_list, all_gather_list, ag_buffer)
                for n in all_gather_dep_list + [merged_all_gather, merged_ag_wait]:
                    if n not in result_list:
                        result_list.append(n)
                compute_list = [i for i in compute_list if i not in result_list]
                result_list.extend(compute_list)

                # merge reduce_scatter
                if stage == "backward" and len(reduce_scatter_list) > 0:
                    (merged_reduce_scatter, rs_buffer, copy_in_size) = merge_reducescatter(sched, reduce_scatter_list)
                    merged_rs_wait = merge_rs_wait(sched, rs_wait_list, reduce_scatter_list, rs_buffer, copy_in_size)

                    for n in [merged_reduce_scatter, merged_rs_wait] + rs_wait_dep_list:
                        if n not in result_list:
                            result_list.append(n)

                # clear the list for bucketing
                all_gather_list = [node]
                all_gather_dep_list = ag_info_dict[node].AG_INV_DEP
                ag_wait_list = ag_info_dict[node].AG_WAIT
                compute_list = ag_info_dict[node].COMPUTE

                # clear the number for greedy
                current_comp = next_comp
                current_ag = ag_info_dict[node].AG_TIME 
                current_mem = ag_info_dict[node].COMPUTE_MEMORY
                next_comp = ag_info_dict[node].COMPUTE_TIME

                if stage == "backward":
                    reduce_scatter_list = ag_info_dict[node].REDUCE_SCATTER
                    rs_wait_list = ag_info_dict[node].RS_WAIT
                    rs_wait_dep_list = ag_info_dict[node].RS_WAIT_DEP
                    current_rs = next_rs
                    next_rs = ag_info_dict[node].RS_TIME
                    
                # the first AG is not bucketed
                first_AG = False
            else:
                # update the list for bucketing
                all_gather_list.append(node)
                all_gather_dep_list.extend(ag_info_dict[node].AG_INV_DEP)
                ag_wait_list.extend(ag_info_dict[node].AG_WAIT)
                compute_list.extend(ag_info_dict[node].COMPUTE)

                # update the number for greedy
                current_mem += ag_info_dict[node].COMPUTE_MEMORY
                current_ag = estimate_bucked_nccl_collective_runtime(all_gather_list)
                next_comp += ag_info_dict[node].COMPUTE_TIME

                if stage == "backward":
                    reduce_scatter_list.extend(ag_info_dict[node].REDUCE_SCATTER)
                    rs_wait_list.extend(ag_info_dict[node].RS_WAIT)
                    rs_wait_dep_list.extend(ag_info_dict[node].RS_WAIT_DEP)
                    if len(reduce_scatter_list) > 0:
                        next_rs = estimate_bucked_nccl_collective_runtime(reduce_scatter_list)

    if len(all_gather_list) > 0:
        merged_all_gather, ag_buffer = merge_allgather(sched, all_gather_list)
        merged_ag_wait = merge_ag_wait(sched, ag_wait_list, all_gather_list, ag_buffer)
        for n in all_gather_dep_list + [merged_all_gather, merged_ag_wait]:
            if n not in result_list:
                result_list.append(n)
    compute_list = [i for i in compute_list if i not in result_list]
    result_list.extend(compute_list)

    # merge reduce_scatter
    if stage == "backward" and len(reduce_scatter_list) > 0:
        (merged_reduce_scatter, rs_buffer, copy_in_size) = merge_reducescatter(sched, reduce_scatter_list)
        merged_rs_wait = merge_rs_wait(sched, rs_wait_list, reduce_scatter_list, rs_buffer, copy_in_size)    
        
        for n in [merged_reduce_scatter, merged_rs_wait] + rs_wait_dep_list:
            if n not in result_list:
                result_list.append(n)
    return result_list


def bucket_forward(
    sched: "scheduler.Scheduler",
    snodes: List["scheduler.BaseSchedulerNode"],
    run_time_dict: Dict[str, List[Union[str, float, float]]],
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Greedy bucket ALL_GATHER and AG_WAIT
    """
    ag_info_dict = get_ag_info(snodes, run_time_dict)
    result_list = get_greedy_bucket_plan(sched, snodes, ag_info_dict)
    return result_list


def bucket_backward(
    sched: "scheduler.Scheduler",
    snodes: List["scheduler.BaseSchedulerNode"],
    run_time_dict: Dict[str, List[Union[str, float, float]]],
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Greedy bucket REDUCE_SCATTER and RS_WAIT
    """
    ag_info_dict = get_ag_info(snodes, run_time_dict, stage="backward")
    result_list = get_greedy_bucket_plan(sched, snodes, ag_info_dict, stage="backward")
    return result_list
