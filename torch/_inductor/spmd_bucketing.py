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

import torch.distributed._functional_collectives
_c10d_functional = torch.ops._c10d_functional

def check_block_interfuse(
    snodes: List["scheduler.BaseSchedulerNode"],
    print_node_origin: Optional[bool] = False,
) -> List["scheduler.BaseSchedulerNode"]:
    count = 0
    if print_node_origin:
        print("***********print node origin *************")
        count = 0
        for node in snodes:
            if isinstance(node, scheduler.SchedulerNode):
                print("Comm", type(node.node) == ir._CollectiveKernel, "| Wait", type(node.node) == ir._WaitKernel, "| origins: ", node.node.origins, "| origin_node: ", node.node.origin_node)
            elif isinstance(node, scheduler.FusedSchedulerNode):
                for subnode in node.snodes:
                    print("Comm", type(node.node) == ir._CollectiveKernel, "| Wait", type(node.node) == ir._WaitKernel, "| origins: ", subnode.node.origins, "| origin_node: ", subnode.node.origin_node)
            elif isinstance(node, scheduler.ExternKernelSchedulerNode):
                print("Comm", type(node.node) == ir._CollectiveKernel, "| Wait", type(node.node) == ir._WaitKernel, "| origins: ", node.node.origins, "| origin_node: ", node.node.origin_node)
            elif isinstance(node, scheduler.NopKernelSchedulerNode):
                print("Comm", type(node.node) == ir._CollectiveKernel, "| Wait", type(node.node) == ir._WaitKernel, "| origins: ", node.node.origins, "| origin_node: ", node.node.origin_node)
            count += 1

    for node in snodes:
        if type(node.node) != ir._CollectiveKernel and type(node.node) != ir._WaitKernel:
            if isinstance(node, scheduler.SchedulerNode):
                list_subnode = []
                list_subnode += node.node.origins
                for subnode in list_subnode:
                    print("Name", subnode.name, subnode.meta.get("nn_module_stack", {}))
        '''
        if type(node.node) == ir._CollectiveKernel:
            print(node, node.node)
            origins = []
            origins += node.node.origins
            for subnode in origins:
                if "all_gather_into_tensor" in str(subnode.name):
                    print("origins all gather", subnode.meta)
                if "reduce_scatter_tensor" in str(subnode.name):
                    print("origins reduce scatter", subnode.meta)
            if node.node.origin_node is not None:
                print("node.origin_node", node.node.origin_node.meta)
        '''

def merge_allgather(
    snodes: List["scheduler.BaseSchedulerNode"],
) -> List["scheduler.BaseSchedulerNode"]:
    all_gather_cand = []
    for node in snodes:
        if type(node.node) == ir._CollectiveKernel:
            list_subnode = []
            list_subnode += node.node.origins
            #print(list_subnode)
            for subnode in list_subnode:
                if "all_gather_into_tensor" in str(subnode.name):
                    all_gather_cand.append(subnode)
                    #print(list_subnode[0], list_subnode[0].meta)
                    #print(list_subnode[1], list_subnode[1].meta)
                     
            if len(all_gather_cand) == 2:
                break
        if type(node.node) == ir._WaitKernel:
            list_subnode = []
            list_subnode += node.node.origins
            #print(list_subnode)
    #import pdb; pdb.set_trace()
    for ag in all_gather_cand:
        print("***********************")
        static = ag.__getstate__()
        print(list(static.keys()))
        print(static)

def bucketing_per_blcok(
    snodes: List["scheduler.BaseSchedulerNode"],
    print_node_origin: Optional[bool] = False,
) -> List["scheduler.BaseSchedulerNode"]:

    for node in snodes:
        node_origin_list = []
        if isinstance(node, scheduler.FusedSchedulerNode):
            for subnode in node.snodes:
                node_origin_list += subnode.node.origins
            for n in node_origin_list:
                module_stack = n.meta.get("nn_module_stack", {})
                print("node", node, module_stack)
        else:
            node_origin_list += node.node.origins
            for n in node_origin_list:
                module_stack = n.meta.get("nn_module_stack", {})
                print("node", node, module_stack)
                import pdb;pdb.set_trace()
        
    return snodes
