import math
from enum import IntEnum
from typing import defaultdict, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
from .. import dependencies, ir, scheduler
from ..virtualized import V

def create_scheduler_node_from_ir_node(
    sched: "scheduler.Scheduler",
    node: ir.Operation
) -> "scheduler.BaseSchedulerNode":
    """
    Create a scheduler node from an IR node & setup dependencies
    """
    snode = sched.create_scheduler_node(node)
    snode.min_order = 0
    snode.max_order = 0
    sched.name_to_buf.update({buf.get_name(): buf for buf in snode.get_outputs()})
    sched.name_to_fused_node[snode.get_name()] = snode
    return snode

    
def merge_allgather(
    sched: "scheduler.Scheduler",
    nodes: List["scheduler.BaseSchedulerNode"], 
    dep_nodes: List["scheduler.BaseSchedulerNode"]
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Bucket small ALL_GATHER nodes into one big all_gather node
    the all_gather and its dependencies are returned in a reverse manner
    bc we bucket all_gather reversely
    """
    if len(nodes) == 1:
        return nodes + dep_nodes

    copy_in_inputs = []
    for dep in dep_nodes:
        copy_in_inputs.append(dep.node)

    # create all_gather copy_in's nodes
    inp_split_flatten = [math.prod(cbuf.get_layout().size) for cbuf in copy_in_inputs]
    inp_split_sizes = inp_split_flatten
    all_gather_input_numel = sum(inp_split_flatten)
    dtype = torch.bfloat16
    device = torch.device("cuda")
    copy_in_output, _ = ir.FallbackKernel.create(
        torch.ops.fsdp.all_gather_copy_in.default,
        copy_in_inputs,
        inp_split_sizes,
        all_gather_input_numel,
        nodes[0].node.constant_args[0],
        int(nodes[0].node.constant_args[1]),
        dtype,
        device,
        simplefsdp=True,
    )
    copy_in_snode = create_scheduler_node_from_ir_node(sched, V.graph.operations[-3])
    copy_in_output_snode = create_scheduler_node_from_ir_node(
        sched, V.graph.operations[-2]
    )

    # create all_gather's node
    ag_node = ir._CollectiveKernel.create_out_of_place(
        torch.ops._c10d_functional.all_gather_into_tensor.default,
        copy_in_output,
        nodes[0].node.constant_args[0],
        nodes[0].node.constant_args[1],
    ) 
    ag_snode = create_scheduler_node_from_ir_node(sched, ag_node)
    aggregated_nodes = [ag_snode, copy_in_output_snode, copy_in_snode] + dep_nodes
    return aggregated_nodes

def merge_ag_wait(
    sched: "scheduler.Scheduler",
    original_wait_list: List["scheduler.BaseSchedulerNode"],
    original_all_gather_list: List["scheduler.BaseSchedulerNode"],
    agg_node: ir.Operation
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Bucket small AG_WAIT nodes into one big AG_WAIT node
    the ag_wait and its dependencies are returned in a reverse manner
    bc we bucket all_gather reversely
    """
    if len(original_wait_list) == 1:
        return original_wait_list

    # create ag_wait's node
    wait_node = ir._WaitKernel.create_wait(
        torch.ops._c10d_functional.wait_tensor.default,
        ir.TensorBox(ir.StorageBox(agg_node)),
    )
    wait_snode = create_scheduler_node_from_ir_node(sched, wait_node)

    # create ag_wait copy_out's node
    copy_out = ir.FallbackKernel.create(
        torch.ops.fsdp.split_with_sizes_copy.default,
        agg_node,
        tuple(math.prod(n.node.get_layout().size) for n in original_all_gather_list),
        dim=1,
        out=[n.node for n in original_all_gather_list],
        simplefsdp=True,
    )
    copy_out_snode = create_scheduler_node_from_ir_node(sched, copy_out)
    return [copy_out_snode, wait_snode]


def merge_reducescatter(
    sched: "scheduler.Scheduler",
    nodes: List["scheduler.BaseSchedulerNode"], 
    dep_nodes: List["scheduler.BaseSchedulerNode"], 
) -> Tuple[List["scheduler.BaseSchedulerNode"], List[Union[List[int], List[int]]]]:
    """
    Bucket small REDUCE_SCATTER nodes into one big REDUCE_SCATTER node
    """
    if len(nodes) == 1:
        return (dep_nodes + nodes, [])

    # create reduce_scatter copy_in's node
    copy_in_inputs = []
    size = []
    split = [0]
    for dep in dep_nodes:
        copy_in_inputs.append(dep.node)
        layout_size = dep.node.get_layout().size
        split.append(split[-1] + len(layout_size))
        size.extend(layout_size)
    copy_in_size = [size, split]

    copy_in_node = ir.FallbackKernel.create(
        torch.ops.fsdp.chunk_cat.default,
        copy_in_inputs,
        dim=0,
        num_chunks=dist.get_world_size(),
        simplefsdp=True,
    )
    copy_in_snode = create_scheduler_node_from_ir_node(sched, V.graph.operations[-2])
    copy_in_output_snode = create_scheduler_node_from_ir_node(sched, V.graph.operations[-1])
    
    # create reduce_scatter's node
    rs_node = ir._CollectiveKernel.create_out_of_place(
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        copy_in_node,
        nodes[0].node.constant_args[0],
        nodes[0].node.constant_args[1],
        nodes[0].node.constant_args[2],
    )   
    rs_snode = create_scheduler_node_from_ir_node(sched, rs_node)
    aggregated_nodes = dep_nodes + [copy_in_snode, copy_in_output_snode, rs_snode]
    return (aggregated_nodes, copy_in_size)


def merge_rs_wait(
    sched: "scheduler.Scheduler",
    original_wait_list: List["scheduler.BaseSchedulerNode"],
    wait_dep_list: List["scheduler.BaseSchedulerNode"],
    rs_node: ir.Operation,
    copy_in_size: List[Union[List[int], List[int]]],
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Bucket small RS_WAIT nodes into one big RS_WAIT node
    """
    if len(original_wait_list) == 1:
        return original_wait_list + wait_dep_list

    # create rs_wait's node
    wait_node = ir._WaitKernel.create_wait(
        torch.ops._c10d_functional.wait_tensor.default,
        ir.TensorBox(ir.StorageBox(rs_node)),
    )
    wait_snode = create_scheduler_node_from_ir_node(sched, wait_node)

    # create rs_wait copy_out's node
    copy_out = ir.FallbackKernel.create(
        torch.ops.fsdp.read_out.default,
        rs_node,
        size=copy_in_size[0],
        split=copy_in_size[1],
        dim=1,
        out=[n.node for n in wait_dep_list],
    )
    copy_out = create_scheduler_node_from_ir_node(sched, copy_out)
    return [wait_snode, copy_out]
