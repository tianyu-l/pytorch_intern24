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

from . import config, ir, scheduler
from collections import OrderedDict

import torch

from . import config, ir, scheduler
from .dependencies import WeakDep
from .utils import is_collective, is_wait, tuple_sorted

def reorder_manual(
    snodes: List["scheduler.BaseSchedulerNode"],
    print_node_origin: Optional[bool] = True,
    reorder_forward: Optional[bool] = False,
) -> List["scheduler.BaseSchedulerNode"]:
    print("snodes", len(snodes))
    if len(snodes) == 234:
        print("************************* forward node after fusion *************************")
        print("originla node")
        print(snodes)
    if reorder_forward and len(snodes) == 155: # forward graph node num
        print("************************* try reorder forward node *************************")
        comm1 = snodes[:6]
        comm1_wait = snodes[6:8]
        comp1 = [snodes[8]]
        comm2 = snodes[9:11]
        comm2_wait = snodes[11:13]
        comp2 = [snodes[13]]
        comm3 = snodes[14:16]
        comm3_wait = snodes[16:18]
        comp3 = [snodes[18]]
        comm4 = snodes[19:21]
        comm4_wait = snodes[21:23]
        comp4 = snodes[23:49]
        comm5 = snodes[49:51]
        comm5_wait = snodes[51:53]
        snodes = comm1 + comm1_wait + comm2 + comp1 + comm2_wait + comm3 + comp2 + comm3_wait + comm4 + comp3 + comm4_wait + comm5 + comp4 + comm5_wait + snodes[53:]
        #snodes = comm1_node+comm2_node+comm3_node+comm4_node+comm5_node+[comp1_node]+[comp2_node]+[comp3_node]+comp4_node+ snodes[53:]
        print("reordered node")
        print(snodes)
    #if len(snodes) == 234:
    print("************************* print the graph mappings *************************")
    if print_node_origin:
        count = 0
        for node in snodes:
            if isinstance(node, scheduler.SchedulerNode):
                print("Node index", count, "| SchedulerNode node name: ", node.node.name, "| fx IR original node: ", node.node.origins, "| Node type: ", type(node.node), node.inverse_users)
            elif isinstance(node, scheduler.FusedSchedulerNode):
                for subnode in node.snodes:
                    print("Node index", count, "| FusedSchedulerNode subnode name: ", subnode.node.name, "| fx IR original node: ", subnode.node.origins, "| Node type: ", type(subnode.node), subnode.inverse_users)
            elif isinstance(node, scheduler.ExternKernelSchedulerNode):
                print("Node index", count, "| ExternKernelSchedulerNode node name: ", node.node.name, "| fx IR original node: ", node.node.origins, "| Node type: ", type(node.node), node.inverse_users)
            elif isinstance(node, scheduler.NopKernelSchedulerNode):
                print("Node index", count, "| NopKernelSchedulerNode node name: ", node.node.name, "| fx IR original node: ", node.node.origins, "| Node type: ", type(node.node), node.inverse_users)
            count += 1
    return snodes


def reorder_compute_heuristic(
    snodes: List["scheduler.BaseSchedulerNode"],
    print_node_origin: Optional[bool] = False,

) -> List["scheduler.BaseSchedulerNode"]:
    if print_node_origin:
        count = 0
        for node in snodes:
            if isinstance(node, scheduler.SchedulerNode):
                print("Node index", count, "| SchedulerNode node name: ", node.node.name, "| fx IR original node: ", node.node.origins, "| Node type: ", type(node.node))
            elif isinstance(node, scheduler.FusedSchedulerNode):
                for subnode in node.snodes:
                    print("Node index", count, "| FusedSchedulerNode subnode name: ", subnode.node.name, "| fx IR original node: ", subnode.node.origins, "| Node type: ", type(subnode.node))
            elif isinstance(node, scheduler.ExternKernelSchedulerNode):
                print("Node index", count, "| ExternKernelSchedulerNode node name: ", node.node.name, "| fx IR original node: ", node.node.origins, "| Node type: ", type(node.node))
            elif isinstance(node, scheduler.NopKernelSchedulerNode):
                print("Node index", count, "| NopKernelSchedulerNode node name: ", node.node.name, "| fx IR original node: ", node.node.origins, "| Node type: ", type(node.node))
            count += 1
    # tag each node
    # node_tag: {0: all gather; 1: wait_tensor; 2: computation}
    # each node has a sublist, [compute_step, node_tag]
    node_tags = []
    compute_step = 0
    for _, node in enumerate(snodes):
        if type(node.node) == ir._CollectiveKernel:
            compute_step = compute_step+1
            node_tags.append([compute_step, 0])
        elif type(node.node) == ir._WaitKernel:
            node_tags.append([compute_step, 1])
        else:
            node_tags.append([compute_step, 2])

    fuse_node_mapping = {}
    for node in snodes:
        if isinstance(node, scheduler.FusedSchedulerNode):
            for subnode in node.snodes:
                fuse_node_mapping[subnode] = node
    fuse_node_mapping_key = list(fuse_node_mapping.keys())
    def get_dependency(node, cur_list, dependecy_list=[]):
        # get node dependency
        dependent = node.inverse_users # the fused nodes are
        #import pdb;pdb.set_trace()
        miss_dependent = [i for i in dependent if i not in cur_list and i not in dependecy_list]
        if dependent in cur_list or len(miss_dependent) == 0:
            return True, []
        else:
            #dependecy_list.extend(miss_dependent)
            for node in miss_dependent:
                dependecy_list.append(node)
                status, sub_dependent = get_dependency(node, cur_list, dependecy_list)
                if not status:
                    for i in sub_dependent:
                        if i not in dependecy_list and i not in cur_list:
                            dependecy_list.append(i)
            return False, dependecy_list

    # reoder the node list
    #print(snodes, len(snodes))
    #print(node_tags)
    reorder_list = []
    cur_compute = []
    start_wait = False
    for idx, node_tag in enumerate(node_tags):
        if node_tag[1] == 0: # gather
            status, dependent = get_dependency(snodes[idx], reorder_list, [])
            if not status:
                dependent.reverse()
                dependent = [fuse_node_mapping[n] if n in fuse_node_mapping_key else n for n in dependent]
                dependent = list(OrderedDict.fromkeys(dependent))
                dependent = [i for i in dependent if i not in reorder_list]
                reorder_list.extend(dependent)
                #print("add dependent", dependent)
            #print("add gather node", snodes[idx], node_tag[0])
            reorder_list.append(snodes[idx])
            start_wait = False
        elif node_tag[1] == 1: # wait
            if not start_wait and len(cur_compute) > 0:
                #print("before pop", cur_compute)
                compute = [i for i in cur_compute if i not in reorder_list]
                #print("after pop", compute)
                #print("add compute node", compute, node_tag[0])
                reorder_list.extend(compute)
                cur_compute = []
           # print("add wait node", snodes[idx], node_tag[0])
            reorder_list.append(snodes[idx])
            start_wait = True
        else: # compute
            cur_compute.append(snodes[idx])
    if len(cur_compute) > 0:
        remain_compute = [i for i in cur_compute if i not in reorder_list]
        #print("add remain compute", remain_compute)
        reorder_list.extend(remain_compute)
   # print("******* reorder_list *******")
    #print(reorder_list, len(reorder_list))

    return reorder_list


def reorder_forward_heuristic(
    snodes: List["scheduler.BaseSchedulerNode"],
    all_gather_order: Optional[str] = "before",
    print_node_origin: Optional[bool] = False,
) -> List["scheduler.BaseSchedulerNode"]:
    #print("start print*******")
    if print_node_origin:
        count = 0
        for node in snodes:
            if isinstance(node, scheduler.SchedulerNode):
                print("Node index", count, "| SchedulerNode node name: ", node.node.name, "| fx IR original node: ", node.node.origins, "| Node dependent: ", node.inverse_users)
            elif isinstance(node, scheduler.FusedSchedulerNode):
                for subnode in node.snodes:
                    print("Node index", count, "| FusedSchedulerNode subnode name: ", subnode.node.name, "| fx IR original node: ", subnode.node.origins, "| Node dependent: ", node.inverse_users)
            elif isinstance(node, scheduler.ExternKernelSchedulerNode):
                print("Node index", count, "| ExternKernelSchedulerNode node name: ", node.node.name, "| fx IR original node: ", node.node.origins, "| Node dependent: ", node.inverse_users)
            elif isinstance(node, scheduler.NopKernelSchedulerNode):
                print("Node index", count, "| NopKernelSchedulerNode node name: ", node.node.name, "| fx IR original node: ", node.node.origins, "| Node dependent: ", node.inverse_users)
            count += 1

    reorder_list = []
    all_gather_list = []
    node_type_list = []
    # node_type: {0: all gather; 1: wait_tensor; 2: computation; 3: reduce scatter}
    for node in reversed(snodes):
        node_type = 2
        if not isinstance(node, scheduler.FusedSchedulerNode):
            node_origins = list(node.node.origins)
            node_origins = [str(i.name) for i in node_origins]
            for name in node_origins:
                # ruisi: all gather and wait won't be in the same node source
                # but do we need to check, just in case?
                if "all_gather_into_tensor" in name:
                    node_type = 0
                if "wait_tensor" in name:
                    node_type = 1
                if "reduce_scatter_tensor" in name:
                    node_type = 3
        node_type_list.append(node_type)
    
    #print(node_type_list)
    for idx, node in enumerate(reversed(snodes)):
        node_type = node_type_list[idx]
        if node_type >= 2:
            if node not in reorder_list and node not in all_gather_list:
                # skip node already in reorder_list
                reorder_list.append(node)
        elif node_type == 1:
            if all_gather_order == "after" and node_type_list[idx+1]==1 and node_type_list[idx+2]==0:
                if len(all_gather_list) > 0:
                    reorder_list.extend(all_gather_list)
                    all_gather_list = []
            reorder_list.append(node)
            if all_gather_order == "before"  and node_type_list[idx-1]==1 and node_type_list[idx+1]==0: # remove this if we only have one wait_tensor
                if len(all_gather_list) > 0:
                    reorder_list.extend(all_gather_list)
                    all_gather_list = []
        else:
            all_gather_list.append(node)
            if len(node.inverse_users) > 0:
                all_gather_list.extend(node.inverse_users)
    if len(all_gather_list) > 0:
        reorder_list.extend(all_gather_list)

    reorder_list.reverse()
    node_type_list.reverse()
    #print("reorder snodes", reorder_list)
    return reorder_list




def reorder_backward_heuristic(
    snodes: List["scheduler.BaseSchedulerNode"],
    all_gather_order: Optional[str] = "before",
    mp_fixed: Optional[bool] = False,
    print_node_origin: Optional[bool] = False,
) -> List["scheduler.BaseSchedulerNode"]:
    reorder_list = []
    wait_list = []
    node_type_list = []
    # node_type: {0: all gather; 1: wait_tensor; 2: computation; 3: reduce scatter; 4: convert_element_type after rs}
    for node in snodes:
        node_type = 2
        if not isinstance(node, scheduler.FusedSchedulerNode):
            node_origins = list(node.node.origins)
            node_origins = [str(i.name) for i in node_origins]
            for name in node_origins:
                # ruisi: all gather and wait won't be in the same node source
                # but do we need to check, just in case?
                if "all_gather_into_tensor" in name:
                    node_type = 0
                if "wait_tensor" in name:
                    node_type = 1
                if "reduce_scatter_tensor" in name:
                    node_type = 3
                if not mp_fixed:
                    if "convert_element_type" in name and node_type_list[-3:] == [3, 1, 1]:
                        node_type = 4
        node_type_list.append(node_type)
    #print(node_type_list)
    if mp_fixed:
        for idx, node in enumerate(snodes):
            node_type = node_type_list[idx]
            if node_type == 1 and ((node_type_list[idx-1] == 3) or (node_type_list[idx-2] == 3 and node_type_list[idx-1] == 1)):
                wait_list.append(node)
            elif node_type == 3:
                if len(wait_list) > 0:
                    reorder_list.extend(wait_list)
                    wait_list = []
                reorder_list.append(node)
            else:
                reorder_list.append(node)
    else:
        for idx, node in enumerate(snodes):
            node_type = node_type_list[idx]
            if node_type == 1 and ((node_type_list[idx-1] == 3) or (node_type_list[idx-2] == 3 and node_type_list[idx-1] == 1)):  
                wait_list.append(node)
            elif node_type == 3:
                if len(wait_list) > 0:
                    #print("add wait node", wait_list)
                    reorder_list.extend(wait_list)
                    wait_list = []
                #print("add rs node", node)
                reorder_list.append(node)
            elif node_type == 4:
                wait_list.append(node)
            else:
                #print("add other node", node)
                reorder_list.append(node)
    if len(wait_list) > 0:
        reorder_list.extend(wait_list)
    return reorder_list
