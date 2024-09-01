import math
from enum import IntEnum
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
from .. import dependencies, ir, scheduler, config

from .utils import NodeType, compute_node_users, get_node_type
from .bucket import merge_allgather, merge_ag_wait, merge_reducescatter, merge_rs_wait


def bucket_forward(
    sched: "scheduler.Scheduler",
    snodes: List["scheduler.BaseSchedulerNode"],
    run_time_dict: Dict[str, List[Union[str, float, float]]],
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Greedy bucket ALL_GATHER and AG_WAIT
    """
    node_dep_dict = {}

    inverse_users, node_users = compute_node_users(snodes)
    comp = []
    skip_nodes = []
    all_gather = None
    total_dep_list = []
    for node in snodes:
        _, run_time, memory = run_time_dict[node.get_name()]

        if get_node_type(node) == NodeType.ALL_GATHER:
            # A ag_dict consists of 6 parts:
            # [Node]: (1) AG_DEP: The node ALL_GATHER depends on for read-in; (2) AG_WAIT: ALL_GATHER's wait nodes; (3) COMPUTE: COMPUTE nodes ALL_GATHER fetches.
            # [Estimation Num.]: (1) AG_TIME: The estimated ALL_GATHER time; (2) COMPUTE_TIME: The estimated COMPUTE time; (3) MEMORY: The estimated meory for computation.
            ag_dict = forward_ag_dict()
            ag_wait = list(node_users[node])
            assert [get_node_type(i) == NodeType.AG_WAIT for i in ag_wait] 
            
            ag_dict["AG_WAIT"] = ag_wait
            ag_dict["AG_TIME"] = run_time * 2
            node_dep_dict[node] = ag_dict
            all_gather = node

            if len(skip_nodes) > 0:
                node_dep_dict[all_gather]["AG_DEP"].extend(skip_nodes)
                skip_nodes = []
            dep_list = list(inverse_users[node])
            node_dep_dict[all_gather]["AG_DEP"].extend(dep_list)
            
        if get_node_type(node) == NodeType.COMPUTE:
            users = list(node_users[node])
            users_type = [get_node_type(i) for i in users]
            if NodeType.ALL_GATHER in users_type:
                continue
            if all_gather is None:
                skip_nodes.append(node)
                continue
            
            node_dep_dict[all_gather]["COMPUTE"].append(node)
            node_dep_dict[all_gather]["COMPUTE_TIME"] += run_time
            node_dep_dict[all_gather]["MEMORY"] += memory
    # make sure all nodes are indexed in node_dep_dict
    count = 0
    for key, value in node_dep_dict.items():
        for k, v in value.items():
            if  k == "AG_WAIT" or k == "COMPUTE" or k == "AG_DEP":
                count += len(v)
        count = count+1
    print("count", count, "snodes", len(snodes))
    #assert count == len(snodes)
    
    result_list = []
    all_gather_list = []
    all_gather_dep_list = []
    all_gather_dep_aux_list = []
    ag_wait_list = []
    compute_list = []
    current_comp = 0 # compute time in step_i
    current_ag = 0 # all gather time in step_i
    current_mem = 0 # memory in step_i
    next_comp = 0 # compute time in step_(i+1), derived from all gather in step_i
    first_AG = True

    for idx, node in enumerate(snodes):
        if get_node_type(node) == NodeType.ALL_GATHER: 
            enable_bucket = forward_greedy(current_comp, current_ag, current_mem, 
                node_dep_dict[node]["COMPUTE_TIME"], node_dep_dict[node]["AG_TIME"], 
                node_dep_dict[node]["MEMORY"], 8e8, first_AG)
            first_AG = False
            if enable_bucket or len(all_gather_list) >= config.simplefsdp.max_greedy_bucket_size:
                print("enable bucket1", len(all_gather_list), enable_bucket, len(ag_wait_list))
                # merge all gather
                merged_all_gather = merge_allgather(sched, all_gather_list)
                merged_wait = merge_ag_wait(sched, ag_wait_list, all_gather_list, merged_all_gather[-1].node)
                for n in all_gather_dep_list + merged_all_gather + merged_wait:
                    if n not in result_list:
                        result_list.append(n)
                compute_list = [i for i in compute_list if i not in result_list]
                result_list.extend(compute_list)

                # clear the list for bucketing
                all_gather_list = [node]
                all_gather_dep_list = node_dep_dict[node]["AG_DEP"]
                ag_wait_list = node_dep_dict[node]["AG_WAIT"]
                compute_list = node_dep_dict[node]["COMPUTE"]

                # clear the number for greedy
                current_comp = next_comp
                current_mem = node_dep_dict[node]["MEMORY"]
                current_ag = node_dep_dict[node]["AG_TIME"] 
                next_comp = node_dep_dict[node]["COMPUTE_TIME"]
            else:
                # update the list for bucketing
                all_gather_list.append(node)
                all_gather_dep_list.extend(node_dep_dict[node]["AG_DEP"])
                ag_wait_list.extend(node_dep_dict[node]["AG_WAIT"])
                compute_list.extend(node_dep_dict[node]["COMPUTE"])

                # update the number for greedy
                current_mem += node_dep_dict[node]["MEMORY"]
                current_ag += node_dep_dict[node]["AG_TIME"]
                next_comp += node_dep_dict[node]["COMPUTE_TIME"]
    
    if len(all_gather_list) > 0:
        print("enable bucket2", len(all_gather_list), len(all_gather_dep_list), enable_bucket, len(ag_wait_list))
        merged_all_gather = merge_allgather(sched, all_gather_list)
        merged_wait = merge_ag_wait(sched, ag_wait_list, all_gather_list, merged_all_gather[-1].node)
        for n in all_gather_dep_list + merged_all_gather + merged_wait:
            if n not in result_list:
                result_list.append(n)
        compute_list = [i for i in compute_list if i not in result_list]
        result_list.extend(compute_list)
   
    return result_list


def forward_greedy(current_comp, current_ag, current_mem, node_comp, node_ag, node_mem, memory_constraint, first_AG):
    """
    Greedy algorithm for forward
    """
    enable_bucket = False

    if first_AG:
        enable_bucket = True

    if current_mem + node_mem > memory_constraint:
        print("memory constraint not satisfied")
        enable_bucket = True

    if current_ag + node_ag > current_comp:
        print("time constraint not satisfied")
        enable_bucket = True
    
    if current_ag == 0:
        enable_bucket = False
    
    return enable_bucket


def forward_ag_dict():
    return {
        "AG_DEP_AUX": [],
        "AG_DEP": [],
        "AG_WAIT": [],
        "COMPUTE": [],
        "AG_TIME": 0,
        "COMPUTE_TIME": 0,
        "MEMORY": 0,
    }

def bucket_backward(
    sched: "scheduler.Scheduler",
    snodes: List["scheduler.BaseSchedulerNode"],
    run_time_dict: Dict[str, List[Union[str, float, float]]],
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Greedy bucket REDUCE_SCATTER and RS_WAIT
    """
    # estimate the run time of each node
    node_dep_dict = {}
    inverse_users, node_users = compute_node_users(snodes)
    all_gather = None
    skip_nodes = []

    for node in snodes:
        _, run_time, memory = run_time_dict[node.get_name()]

        if get_node_type(node) == NodeType.ALL_GATHER:
            # A node_user_dict consists of 9 parts:
            # [Node]: (1) AG_DEP: The node ALL_GATHER depends on for read-in; (2) AG_WAIT: ALL_GATHER's wait nodes; (3) COMPUTE: COMPUTE nodes ALL_GATHER fetches.
            # （4） REDUCE_SCATTER: REDUCE_SCATTE that reads from COMPUTE nodes; (5) RS_WAIT: REDUCE_SCATTER's wait nodes;
            # [Estimation Num.]: (1) AG_TIME: The estimated ALL_GATHER time; (2) COMPUTE_TIME: The estimated COMPUTE time; (3) MEMORY: The estimated meory for computation.
            # (4) RS_TIME: The estimated REDUCE_SCATTER time; 
            node_user_dict = backward_ag_dict()
            ag_wait = list(node_users[node])
            assert [get_node_type(i) == NodeType.AG_WAIT for i in ag_wait] 

            node_user_dict["AG_WAIT"] = ag_wait
            node_user_dict["AG_TIME"] = run_time * 2
            node_dep_dict[node] = node_user_dict
            all_gather = node

            if len(skip_nodes) > 0:
                node_dep_dict[all_gather]["AG_DEP"].extend(skip_nodes)
                skip_nodes = []
            node_dep_dict[all_gather]["AG_DEP"].extend(list(inverse_users[node]))
            
        if get_node_type(node) == NodeType.COMPUTE:
            users_type = [get_node_type(i) for i in list(node_users[node])]
            inverse_users_type = [get_node_type(i) for i in list(inverse_users[node])]
            if NodeType.ALL_GATHER in users_type:
                continue
            if NodeType.RS_WAIT in inverse_users_type:
                continue
            if all_gather is None:
                skip_nodes.append(node)
            else:
                node_dep_dict[all_gather]["COMPUTE"].append(node)
                node_dep_dict[all_gather]["COMPUTE_TIME"] += run_time
                node_dep_dict[all_gather]["MEMORY"] += memory

        if get_node_type(node) == NodeType.REDUCE_SCATTER:
            # rs_dep --> rs --> rs_wait --> rs_wait_dep
            if all_gather is None:
                skip_nodes.append(node)
            else:
                node_dep_dict[all_gather]["REDUCE_SCATTER"].append(node)
                
        if get_node_type(node) == NodeType.RS_WAIT:
            if all_gather is None:
                skip_nodes.append(node)
            else:
                node_dep_dict[all_gather]["RS_WAIT"].append(node)
                node_dep_dict[all_gather]["RS_WAIT_DEP"].extend(list(node_users[node]))

    count = 0
    for key, value in node_dep_dict.items():
        for k, v in value.items():
            if k == "AG_DEP" or k == "AG_WAIT" or k == "COMPUTE" or k == "REDUCE_SCATTER" or k == "RS_WAIT" or k == "RS_WAIT_DEP":
                count += len(v)
        if get_node_type(key) == NodeType.ALL_GATHER:
            count += 1 
    print("count", count, "snodes", len(snodes))
    #assert count == len(snodes)

    result_list = []
    all_gather_list = []
    all_gather_dep_list = []
    all_gather_dep_aux_list = []
    ag_wait_list = []
    compute_list = []
    reduce_scatter_list = []
    reduce_scatter_dep_list = []
    reduce_scatter_dep_aux_list = []
    rs_wait_list = []
    rs_wait_dep_list = []


    current_comp = 0 # compute time in step_i
    current_ag = 0 # all gather time in step_i
    current_mem = 0 # memory in step_i
    current_rs = 0 # reduce scatter time in step_i
    next_comp = 0 # compute time in step_(i+1), derived from all gather in step_i
    next_rs = 0 # reduce scatter time in step_(i+1), derived from all gather in step_i
    first_AG = True

    #print("node_dep_dict", node_dep_dict)
    for idx, node in enumerate(snodes):
        if get_node_type(node) == NodeType.ALL_GATHER: 
            enable_bucket = backward_greedy(current_comp, current_ag, current_rs, current_mem, 
                node_dep_dict[node]["COMPUTE_TIME"], node_dep_dict[node]["AG_TIME"], 
                node_dep_dict[node]["MEMORY"], node_dep_dict[node]["RS_TIME"], 8e8, first_AG)
            first_AG = False

            if enable_bucket or len(all_gather_list) >= config.simplefsdp.max_greedy_bucket_size or len(reduce_scatter_list) >= config.simplefsdp.max_greedy_bucket_size:
                print("enable bucket", len(all_gather_list), enable_bucket, len(reduce_scatter_list))
                # merge all_gather  
                merged_all_gather = merge_allgather(sched, all_gather_list)
                merged_ag_wait = merge_ag_wait(sched, ag_wait_list, all_gather_list, merged_all_gather[-1].node)
                for n in all_gather_dep_list + merged_all_gather + merged_ag_wait:
                    if n not in result_list:
                        result_list.append(n)
                compute_list = [i for i in compute_list if i not in result_list]
                result_list.extend(compute_list)

                # merge reduce_scatter
                if len(reduce_scatter_list) > 0:
                    (merged_reduce_scatter, copy_in_size) = merge_reducescatter(sched, reduce_scatter_list)
                    merged_rs_wait = merge_rs_wait(sched, rs_wait_list, reduce_scatter_list, merged_reduce_scatter[-1].node, copy_in_size)

                    #print("merged_reduce_scatter", len(merged_reduce_scatter), "merged_rs_wait", len(merged_rs_wait), "reduce_scatter_dep_list", len(reduce_scatter_dep_list), reduce_scatter_dep_list[0] in result_list)
                    #print("reduce_scatter_dep_list", [i.node.get_name() for i in reduce_scatter_dep_list])
                    for n in merged_reduce_scatter + merged_rs_wait + reduce_scatter_dep_list:
                        if n not in result_list:
                            result_list.append(n)

                # clear the list for bucketing
                all_gather_list = [node]
                all_gather_dep_list = node_dep_dict[node]["AG_DEP"]
                ag_wait_list = node_dep_dict[node]["AG_WAIT"]
                compute_list = node_dep_dict[node]["COMPUTE"]
                reduce_scatter_list = node_dep_dict[node]["REDUCE_SCATTER"]
                rs_wait_list = node_dep_dict[node]["RS_WAIT"]
                reduce_scatter_dep_list = node_dep_dict[node]["RS_WAIT_DEP"]

                # clear the number for greedy
                current_comp = next_comp
                current_ag = node_dep_dict[node]["AG_TIME"] 
                current_mem = node_dep_dict[node]["MEMORY"]
                current_rs = next_rs
                next_comp = node_dep_dict[node]["COMPUTE_TIME"]
                next_rs = node_dep_dict[node]["RS_TIME"]
            else:
                # update the list for bucketing
                all_gather_list.append(node)
                all_gather_dep_list.extend(node_dep_dict[node]["AG_DEP"])
                ag_wait_list.extend(node_dep_dict[node]["AG_WAIT"])
                compute_list.extend(node_dep_dict[node]["COMPUTE"])
                reduce_scatter_list.extend(node_dep_dict[node]["REDUCE_SCATTER"])
                rs_wait_list.extend(node_dep_dict[node]["RS_WAIT"])
                reduce_scatter_dep_list.extend(node_dep_dict[node]["RS_WAIT_DEP"])

                # update the number for greedy
                current_mem += node_dep_dict[node]["MEMORY"]
                current_ag += node_dep_dict[node]["AG_TIME"]
                next_comp += node_dep_dict[node]["COMPUTE_TIME"]
                next_rs += node_dep_dict[node]["RS_TIME"]

    if len(all_gather_list) > 0:
        print("enable bucket", len(all_gather_list), enable_bucket, len(reduce_scatter_list))
        merged_all_gather = merge_allgather(sched, all_gather_list)
        merged_ag_wait = merge_ag_wait(sched, ag_wait_list, all_gather_list, merged_all_gather[-1].node)
        for n in all_gather_dep_list + merged_all_gather + merged_ag_wait:
            if n not in result_list:
                result_list.append(n)
        compute_list = [i for i in compute_list if i not in result_list]
        result_list.extend(compute_list)

        # merge reduce_scatter
        if len(reduce_scatter_list) > 0:
            (merged_reduce_scatter, copy_in_size) = merge_reducescatter(sched, reduce_scatter_list)
            merged_rs_wait = merge_rs_wait(sched, rs_wait_list, reduce_scatter_list, merged_reduce_scatter[-1].node, copy_in_size)    
            
            #print("merged_reduce_scatter", len(merged_reduce_scatter), "merged_rs_wait", len(merged_rs_wait), "reduce_scatter_dep_list", len(reduce_scatter_dep_list), reduce_scatter_dep_list[0] in result_list)
            #print("reduce_scatter_dep_list", [i.node.get_name() for i in reduce_scatter_dep_list])
            for n in merged_reduce_scatter + merged_rs_wait + reduce_scatter_dep_list:
                if n not in result_list:
                    result_list.append(n)


    print("backward node", len(result_list), "set node", len(list(set(result_list))))
    return result_list


def backward_greedy(current_comp, current_ag, current_rs, current_mem, node_comp, node_ag, node_mem, node_rs, memory_constraint, first_AG):
    """
    Greedy algorithm for backward
    """
    enable_bucket = False
    if first_AG:
        enable_bucket = True

    if current_mem + node_mem > memory_constraint:
        print("memory constraint not satisfied")
        enable_bucket = True

    if current_ag + node_ag + current_rs > current_comp:
        print("time constraint not satisfied")
        enable_bucket = True
    
    if current_ag == 0:
        enable_bucket = False

    return enable_bucket

def backward_ag_dict():
    return {
        "AG_DEP_AUX": [],
        "AG_DEP": [],
        "AG_WAIT": [],
        "COMPUTE": [],
        "REDUCE_SCATTER": [],
        "REDUCE_SCATTER_DEP": [],
        "RS_WAIT": [],
        "RS_WAIT_DEP": [],
        "AG_TIME": 0,
        "COMPUTE_TIME": 0,
        "MEMORY": 0,
        "RS_TIME": 0,
    }
