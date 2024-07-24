# mypy: allow-untyped-defs
# pyre-strict
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Set, defaultdict
import sympy
import math
import sys
import torch
from . import comms, ir, scheduler, dependencies
import torch.distributed as dist
from .virtualized import V
from .dependencies import WeakDep

class NodeType(IntEnum):
    ALL_GATHER = 0
    WAIT = 1
    COMPUTE = 2
    REDUCE_SCATTER = 3


def compute_bucket_users(
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

    #inverse_users = {
    #    node: {buf_to_snode[dep.name] for dep in node.unmet_dependencies}
    #    for node in snodes
    #}

    # compute node_users
    # TODO: ideally, we should deduplicate .users and .node_users,
    # but currently .users contains extra information that's difficult to
    # extract into a standalone container.
    node_users: Dict[scheduler.BaseSchedulerNode, Set[scheduler.BaseSchedulerNode]] = defaultdict(set)
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

    inverse_users, node_users = compute_bucket_users(snodes)
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

    inverse_users, node_users = compute_bucket_users(snodes)

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
    #elif isinstance(node.node, ir.FallbackKernel):
    #    if node.node.op_overload == torch.ops.fsdp.split_with_sizes_copy.default:
    #        return NodeType.ALL_GATHER
    #    elif node.node.op_overload == torch.ops.fsdp.all_gather_copy_in.default:
    #        return NodeType.WAIT

    return NodeType.COMPUTE

def get_node_type_print(node) -> int:
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
    elif isinstance(node.node, ir.FallbackKernel):
        if node.node.op_overload == torch.ops.fsdp.split_with_sizes_copy.default:
            return NodeType.ALL_GATHER
        elif node.node.op_overload == torch.ops.fsdp.all_gather_copy_in.default:
            return NodeType.WAIT

    return NodeType.COMPUTE

def print_node_type(nodes):
    types = []
    for node in nodes:
        types.append(get_node_type_print(node))
    print("types", types)


import torch.distributed._functional_collectives
_c10d_functional = torch.ops._c10d_functional


def create_scheduler_node_from_ir_node(sched, node):
    snode = sched.create_scheduler_node(node)
    snode.min_order = 0
    snode.max_order = 0
    sched.name_to_buf.update({
        buf.get_name(): buf for buf in snode.get_outputs()
    })
    sched.name_to_fused_node[snode.get_name()] = snode
    #print(f"created new snode: {snode.get_name()}")
    return snode

def bucketing_all_gather_per_blcok(
    sched,
    snodes: List["scheduler.BaseSchedulerNode"],
    graph_id: int,
) -> List["scheduler.BaseSchedulerNode"]:
    inverse_users, node_users = comms.compute_node_users(snodes)
    node_block_list = []
    last_module = None
    for node in snodes:
        if isinstance(node, scheduler.FusedSchedulerNode):
            node_module = get_block_level(node.snodes[0])
        else:
            node_module = get_block_level(node)
        if node_module == -1:
            node_module = last_module
        node_block_list.append(node_module)
        last_module = node_module
    
    for i in range(1, len(node_block_list)-2):
        if node_block_list[i] != node_block_list[i-1] and node_block_list[i] != node_block_list[i+1] and node_block_list[i-1] == node_block_list[i+1]:
            node_block_list[i] = node_block_list[i-1]
    count_ag = 0
    compute = []
    comm = []

    #print("add_dep", add_dep)
    bucket_list = []
    bucket_list_name = []
    all_gather_list = []
    all_gather_dep_list = []
    wait_list = []
    depend_dict = {}
    all_gather_dict = {}

    node_block_list.reverse()
    last_module = node_block_list[0]
    additional_single_dep = []
    for idx, node in enumerate(reversed(snodes)):
        current_module = node_block_list[idx]
        if current_module != last_module:
            if isinstance(node, scheduler.FusedSchedulerNode) or (node.node.get_name() not in bucket_list_name and get_node_type(node) == NodeType.COMPUTE):
                #if node.node.get_name() in list(add_dep.keys()):
                #    node.unmet_dependencies = set(list(node.unmet_dependencies) + list(add_dep[node.node.get_name()]))
                #    node.read_writes.reads = set(list(node.read_writes.reads) + list(add_dep[node.node.get_name()]))
                bucket_list.append(node)
                if isinstance(node, scheduler.FusedSchedulerNode):
                    for n in node.snodes:
                        bucket_list_name.append(n.node.get_name())
                else:
                    bucket_list_name.append(node.node.get_name())

            if len(all_gather_list) > 0:
                merged_all_gather, ag_node, all_gather_input_sizes, dep_dic, ag_dict = merge_allgather(sched, all_gather_list, all_gather_dep_list)
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
                    merged_wait, wait_dict, dep_dic = merge_wait(sched, wait_list, all_gather_list, ag_node, all_gather_input_sizes)
                    all_gather_dict.update(wait_dict)
                    depend_dict.update(dep_dic)
                    additional_single_dep = []
                wait_list = []
                all_gather_list = []
                all_gather_dep_list = []
            else:
                merged_wait = []

            to_merge_list = merged_wait+merged_all_gather
            for n in to_merge_list:
                if isinstance(n, torch._inductor.scheduler.BaseSchedulerNode):
                    if n.node.get_name() not in bucket_list_name:
                        bucket_list.append(n)
                        bucket_list_name.append(n.node.get_name())
                else:
                    if n.get_name() not in bucket_list_name:
                        bucket_list.append(n)
                        bucket_list_name.append(n.get_name())

        if get_node_type(node) == NodeType.ALL_GATHER:
            all_gather_list.append(node)
            inverse_user = list(inverse_users[node])
            if len(inverse_user) > 0:
                all_gather_dep_list.extend(inverse_user)
        elif get_node_type(node) == NodeType.WAIT and "all_gather_into_tensor" in node.node.inputs[0].python_kernel_name:
            wait_list.append(node)
        else:
            if isinstance(node, scheduler.FusedSchedulerNode) or (node.node.get_name() not in bucket_list_name and node not in all_gather_dep_list):
                bucket_list.append(node)
                if isinstance(node, scheduler.FusedSchedulerNode):
                    for n in node.snodes:
                        bucket_list_name.append(n.node.get_name())
                else:
                    bucket_list_name.append(node.node.get_name())

        last_module = current_module
    

    if len(all_gather_list) > 0:
        merged_all_gather, ag_node, all_gather_input_sizes, dep_dic, ag_dict = merge_allgather(sched, all_gather_list, all_gather_dep_list)
        depend_dict.update(dep_dic)
        all_gather_dict.update(ag_dict)
    else:
        merged_all_gather = []
    #print("merged_all_gather", [n.node.get_name() for n in merged_all_gather])
    if len(wait_list) > 0:
        if len(wait_list) == 1:
            merged_wait = wait_list
        else:
            merged_wait, wait_dict, dep_dic = merge_wait(sched, wait_list, all_gather_list, ag_node, all_gather_input_sizes)
            all_gather_dict.update(wait_dict)
            depend_dict.update(dep_dic)
    else:
        merged_wait = []
    
    to_merge_list = merged_wait+merged_all_gather
    for node in to_merge_list:
        if isinstance(node, torch._inductor.scheduler.BaseSchedulerNode):
            if node.node.get_name() not in bucket_list_name:
                bucket_list.append(node)
                bucket_list_name.append(node.node.get_name())
        else:
            if node.get_name() not in bucket_list_name:
                bucket_list.append(node)
                bucket_list_name.append(node.get_name())

    bucket_list.reverse()
    buf_list = []
    
    total_node = []
    for node in bucket_list:
        if not isinstance(node, scheduler.FusedSchedulerNode):
            total_node.append(node.node.get_name())
    return bucket_list, depend_dict, all_gather_dict

def merge_allgather(sched, nodes, dep_nodes):
    if len(nodes) == 1:
        return nodes + dep_nodes, nodes[0], None, {}, {}

    copy_in_inputs = []
    for dep in dep_nodes:
        copy_in_inputs.append(dep.node)
   
    #inp_split_sizes = [cbuf.get_layout().size for cbuf in copy_in_inputs]
    inp_split_flatten = [math.prod(cbuf.get_layout().size) for cbuf in copy_in_inputs]
    inp_split_sizes = inp_split_flatten
    all_gather_input_numel = sum(inp_split_flatten)
    dtype = torch.bfloat16
    device = torch.device("cuda")
    copy_in_output, _ = ir.FallbackKernel.create(torch.ops.fsdp.all_gather_copy_in.default, copy_in_inputs, 
            inp_split_sizes, all_gather_input_numel, nodes[0].node.constant_args[0], int(nodes[0].node.constant_args[1]), dtype, device)
    copy_in_snode = create_scheduler_node_from_ir_node(sched, V.graph.operations[-3])
    copy_in_output_snode = create_scheduler_node_from_ir_node(sched, V.graph.operations[-2])
     
    ag_node = ir._CollectiveKernel.create_out_of_place(
        torch.ops._c10d_functional.all_gather_into_tensor.default, 
        copy_in_output, nodes[0].node.constant_args[0], nodes[0].node.constant_args[1]) # 
    ag_snode = create_scheduler_node_from_ir_node(sched, ag_node)
    
    dep_dict = {}
    all_gather_dict = {}
    dep_dict[copy_in_output.inputs[0].get_name()] = {dependencies.StarDep(dep.node.get_name()) for dep in dep_nodes}
    aggregated_nodes = [ag_snode, copy_in_output_snode, copy_in_snode] + dep_nodes
    
    for node in nodes:
        all_gather_dict[node.node.get_name()] = {ag_node.get_name()}
    return aggregated_nodes, ag_node, inp_split_sizes, dep_dict, all_gather_dict
    
def merge_wait(sched, original_wait_list, original_all_gather_list, agg_node, all_gather_input_sizes):
    if len(original_wait_list) == 1:
        return original_wait_list, {}, {}
    wait_node = ir._WaitKernel.create_wait(torch.ops._c10d_functional.wait_tensor.default, ir.TensorBox(ir.StorageBox(agg_node)))
    wait_snode = create_scheduler_node_from_ir_node(sched, V.graph.operations[-1])
    wait_dict = {}
    for node in original_wait_list:
        wait_dict[node.node.get_name()] = {wait_node.get_name()}

    dtype = torch.bfloat16
    device = torch.device("cuda")
    dep_dict = {}

    copy_out = ir.FallbackKernel.create(torch.ops.fsdp.split_with_sizes_copy.default, agg_node,
            tuple(n.node.example_output.numel() for n in original_all_gather_list), dim=1, out=[n.node for n in original_all_gather_list])
    
    copy_out_snode = create_scheduler_node_from_ir_node(sched, copy_out)
    agg_copy_out = [copy_out_snode] #+ copy_in_dep
    return agg_copy_out+[wait_snode], wait_dict, dep_dict
    

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
