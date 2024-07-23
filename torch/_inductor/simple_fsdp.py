# mypy: allow-untyped-defs
# pyre-strict
from enum import IntEnum
from typing import Dict, List, Optional
import sympy
import math

import torch
from . import comms, ir, scheduler, dependencies
import torch.distributed as dist

class NodeType(IntEnum):
    ALL_GATHER = 0
    WAIT = 1
    COMPUTE = 2
    REDUCE_SCATTER = 3


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

    inverse_users, node_users = comms.compute_node_users(snodes)

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


import torch.distributed._functional_collectives
_c10d_functional = torch.ops._c10d_functional

def bucketing_all_gather_per_blcok(
    snodes: List["scheduler.BaseSchedulerNode"],
    graph_id: int,
) -> List["scheduler.BaseSchedulerNode"]:
    inverse_users, node_users = comms.compute_node_users(snodes)
    node_block_list = []
    last_module = None
    for node in snodes:
        node_module = get_block_level(node)
        #if node.node.get_name() == "buf610" and graph_id == 0:
        #    node_module = last_module
        #print("node_module", node_module)
        if node_module == -1:
            node_module = last_module
        if last_module == "L['self'].output":
            node_module = last_module
        node_block_list.append(node_module)
        last_module = node_module

    count_ag = 0
    compute = []
    comm = []
    add_dep = {}
    for idx, node in enumerate(snodes):
        current_module = node_block_list[idx]
       # if count_ag == 1:
        #    print("node", node.node.get_name(), [c.node.get_name() for c in comm], current_module, last_module, [c.node.get_name() for c in compute])
        if current_module != last_module:
            if count_ag == 1:
                comm = comm + list(inverse_users[comm[0]])
                for c in compute:
                    if c not in comm:
                        add_dep[c.node.get_name()] = set([dependencies.StarDep(com.node.get_name()) for com in comm] )
            count_ag = 0
            compute = []
            comm = []
        if get_node_type(node) == NodeType.ALL_GATHER:
            count_ag += 1
            comm.append(node)
        elif get_node_type(node) == NodeType.WAIT and "all_gather_into_tensor" in node.node.inputs[0].python_kernel_name:
            comm.append(node)
        else:
            compute.append(node)
        last_module = current_module
    #print("comm", count_ag)
    if count_ag == 1:
        comm = comm + list(inverse_users[comm[0]])
        for c in compute:
            if c not in comm:
                add_dep[c.node.get_name()] = set([dependencies.StarDep(com.node.get_name()) for com in comm])
 
    #print("add_dep", add_dep)
    bucket_list = []
    bucket_list_name = []
    all_gather_list = []
    all_gather_dep_list = []
    wait_list = []
    depend_dict = {}
    all_gather_dict = {}
    last_module = node_block_list[0]

    node_block_list.reverse()
    additional_single_dep = []
    for idx, node in enumerate(reversed(snodes)):
        current_module = node_block_list[idx]
        if current_module != last_module:
            if node.node.get_name() not in bucket_list_name and get_node_type(node) == NodeType.COMPUTE:
                if node.node.get_name() in list(add_dep.keys()):
                    node.unmet_dependencies = set(list(node.unmet_dependencies) + list(add_dep[node.node.get_name()]))
                    node.read_writes.reads = set(list(node.read_writes.reads) + list(add_dep[node.node.get_name()]))
                bucket_list.append(node)
                bucket_list_name.append(node.node.get_name())

            if len(all_gather_list) > 0:
                merged_all_gather, ag_node, all_gather_input_sizes, out_split_sizes, dep_dic, ag_dict = merge_allgather(all_gather_list, all_gather_dep_list)
                depend_dict.update(dep_dic)
                all_gather_dict.update(ag_dict)
                if len(all_gather_list) == 1:
                    additional_single_dep.extend(all_gather_list)
                else:
                    additional_single_dep = []
            else:
                merged_all_gather = []
            
            if len(wait_list) > 0:
                if len(wait_list) == 1:
                    merged_wait = wait_list
                    additional_single_dep.extend(wait_list)
                else:
                    assert len(all_gather_list) == len(wait_list)
                    merged_wait, wait_dict, dep_dic = merge_wait(wait_list, all_gather_list, ag_node, all_gather_input_sizes, out_split_sizes)
                    all_gather_dict.update(wait_dict)
                    depend_dict.update(dep_dic)
                    additional_single_dep = []
                wait_list = []
                all_gather_list = []
                all_gather_dep_list = []
            else:
                merged_wait = []

            to_merge_list = merged_wait+merged_all_gather
            for node in to_merge_list:
                if isinstance(node, scheduler.BaseSchedulerNode):
                    if node.node.get_name() not in bucket_list_name:
                        bucket_list.append(node)
                        bucket_list_name.append(node.node.get_name())
                else:
                    if node.get_name() not in bucket_list_name:
                        bucket_list.append(node)
                        bucket_list_name.append(node.get_name())

        if get_node_type(node) == NodeType.ALL_GATHER:
            all_gather_list.append(node)
            inverse_user = list(inverse_users[node])
            if len(inverse_user) > 0:
                all_gather_dep_list.extend(inverse_user)
        elif get_node_type(node) == NodeType.WAIT and "all_gather_into_tensor" in node.node.inputs[0].python_kernel_name:
            wait_list.append(node)
        else:
            if node.node.get_name() not in bucket_list_name:
                if node.node.get_name() in list(add_dep.keys()):
                    node.unmet_dependencies = set(list(node.unmet_dependencies) + list(add_dep[node.node.get_name()]))
                    node.read_writes.reads = set(list(node.read_writes.reads) + list(add_dep[node.node.get_name()]))
                bucket_list.append(node)
                bucket_list_name.append(node.node.get_name())

        last_module = current_module
    

    if len(all_gather_list) > 0:
        merged_all_gather, ag_node, all_gather_input_sizes, out_split_sizes, dep_dic, ag_dict = merge_allgather(all_gather_list, all_gather_dep_list)
        depend_dict.update(dep_dic)
        all_gather_dict.update(ag_dict)
    else:
        merged_all_gather = []
    if len(wait_list) > 0:
        if len(wait_list) == 1:
            merged_wait = wait_list
        else:
            merged_wait, wait_dict, dep_dic = merge_wait(wait_list, all_gather_list, ag_node, all_gather_input_sizes, out_split_sizes)
            all_gather_dict.update(wait_dict)
            depend_dict.update(dep_dic)
    else:
        merged_wait = []
    
    to_merge_list = merged_wait+merged_all_gather
    for node in to_merge_list:
        if isinstance(node, scheduler.BaseSchedulerNode):
            if node.node.get_name() not in bucket_list_name:
                #print("add other node", node.node.get_name())
                bucket_list.append(node)
                bucket_list_name.append(node.node.get_name())
        else:
            if node.get_name() not in bucket_list_name:
                #print("add other node", node.get_name())
                bucket_list.append(node)
                bucket_list_name.append(node.get_name())

    bucket_list.reverse()
    buf_list = []
    
    for node in bucket_list:
        if isinstance(node, scheduler.BaseSchedulerNode):
            buf_list.append(node.node.get_name())
        else:
            buf_list.append(node.get_name())
    #print("buf_list", buf_list)
    return bucket_list, depend_dict, all_gather_dict

def merge_allgather(nodes, dep_nodes):
    print("start to merge", [node.node.get_name() for node in nodes])
    print("start to merge dep", [node.node.get_name() for node in dep_nodes])
    if len(nodes) == 1:
        return nodes + dep_nodes, nodes[0], None, None, {}, {}

    copy_in_inputs = []
    for dep in dep_nodes:
        copy_in_inputs.append(dep.node)
   
    out_split_sizes = [cbuf.get_layout().size for cbuf in copy_in_inputs]
    inp_split_sizes = [math.prod(cbuf.get_layout().size) for cbuf in copy_in_inputs]
    all_gather_input_numel = sum(inp_split_sizes)
    dtype = torch.bfloat16
    device = torch.device("cuda")
    copy_in_ouput, _ = ir.FallbackKernel.create(torch.ops.fsdp.all_gather_copy_in.default, copy_in_inputs, 
            inp_split_sizes, all_gather_input_numel, nodes[0].node.constant_args[0], int(nodes[0].node.constant_args[1]), dtype, device)
    ag_node = ir._CollectiveKernel.create_out_of_place(
        torch.ops._c10d_functional.all_gather_into_tensor.default, 
        copy_in_ouput, nodes[0].node.constant_args[0], nodes[0].node.constant_args[1]) # 
 
    #print("ag_node", ag_node)
    dep_dict = {}
    all_gather_dict = {}
    dep_dict[copy_in_ouput.inputs[0].get_name()] = {dependencies.StarDep(dep.node.get_name()) for dep in dep_nodes}
    
    for node in dep_nodes:
        node.unmet_dependencies = set()
        node.read_writes.reads = set()
    
    aggregated_nodes = [ag_node, copy_in_ouput, copy_in_ouput.inputs[0]] + dep_nodes
    
    for node in nodes:
        all_gather_dict[node.node.get_name()] = {ag_node.get_name()}
    return aggregated_nodes, ag_node, inp_split_sizes, out_split_sizes, dep_dict, all_gather_dict
    
def merge_wait(original_wait_list, original_all_gather_list, agg_node, all_gather_input_sizes, out_split_sizes):
    if len(original_wait_list) == 1:
        return original_wait_list, {}, {}
    wait_node = ir._WaitKernel.create_wait(torch.ops._c10d_functional.wait_tensor.default, ir.TensorBox(ir.StorageBox(agg_node)))
    
    wait_dict = {}
    for node in original_wait_list:
        wait_dict[node.node.get_name()] = {wait_node.get_name()}
    
    dep_nodes = []
    for node in original_wait_list:
        dep_nodes += list(node.read_writes.reads)

    dtype = torch.bfloat16
    device = torch.device("cuda")
    dep_dict = {}

    copy_out = ir.FallbackKernel.create(torch.ops.fsdp.split_with_sizes_copy.default, agg_node,
            tuple(n.node.example_output.numel() for n in original_all_gather_list), dim=0, out=[n.node for n in original_all_gather_list])
    
    dep_dict[node.node.get_name()] =  dep_nodes
    dep_dict[copy_out.get_name()] = {dependencies.StarDep(wait_node.get_name())}
    #set_tensor = ir.FallbackKernel.create(torch.ops._c10d_functional.set_tensor.default, copy_out)
    #print("copy_out", copy_out)
    copy_in_dep = []
    for node in copy_out.inputs[1:]:
        copy_in_dep.append(node)
        dep_dict[node.get_name()] = {dependencies.StarDep(wait_node.get_name())}

    #agg_copy_out = copy_in_dep+[copy_out]
    agg_copy_out = [copy_out] #+ copy_in_dep
    return [wait_node]+agg_copy_out, wait_dict, dep_dict
    

def get_block_level(node):
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
    return -1
