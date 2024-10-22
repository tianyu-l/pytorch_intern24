import re
from typing import List

from .. import scheduler
from .bucket import merge_ag_wait, merge_allgather, merge_reducescatter, merge_rs_wait
from .utils import compute_node_users, get_node_type, NodeType


def bucket_all_gather_by_block(
    sched: "scheduler.Scheduler",
    snodes: List["scheduler.BaseSchedulerNode"],
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Bucket ALL_GATHER and AG_WAIT by block
    """
    inverse_users, node_users = compute_node_users(snodes)

    # get the block each node belongs to
    node_block_list = []
    last_module = ""
    for node in snodes:
        if isinstance(node, scheduler.FusedSchedulerNode) or isinstance(
            node, scheduler.GroupedSchedulerNode
        ):
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
    compute_list = []
    last_module = node_block_list[0]
    for idx, node in enumerate(snodes):
        current_module = node_block_list[idx]
        if current_module != last_module and len(all_gather_list) > 0:
            # bucketing in the block boundary
            assert len(all_gather_list) == len(ag_wait_list)
            merged_all_gather, ag_ir_node = merge_allgather(
                sched, all_gather_list, all_gather_dep_list
            )
            merged_wait = merge_ag_wait(
                sched, ag_wait_list, all_gather_list, ag_ir_node
            )
            for n in [merged_all_gather, merged_wait]:
                if n not in result_list:
                    result_list.append(n)
            compute_list = [i for i in compute_list if i not in all_gather_dep_list]
            result_list.extend(compute_list)
            ag_wait_list = []
            all_gather_list = []
            all_gather_dep_list = []
            compute_list = []

        if get_node_type(node) == NodeType.ALL_GATHER:
            # add the small all_gather to bucket
            all_gather_list.append(node)
            all_gather_dep = list(inverse_users[node])
            # pick up all gather's dependency to ensure if happens before all gather
            if len(all_gather_dep) > 0:
                all_gather_dep_list.extend(all_gather_dep)
        elif get_node_type(node) == NodeType.AG_WAIT:
            # add the small ag_wait to bucket
            ag_wait_list.append(node)
        else:
            # we do not bucket other nodes
            if node not in result_list and node not in all_gather_dep_list:
                compute_list.append(node)

        last_module = current_module

    assert len(all_gather_list) == len(ag_wait_list)

    if len(all_gather_list) > 0:
        merged_all_gather, ag_ir_node = merge_allgather(
            sched, all_gather_list, all_gather_dep_list
        )
        merged_wait = merge_ag_wait(sched, ag_wait_list, all_gather_list, ag_ir_node)
        for n in [merged_all_gather, merged_wait]:
            if n not in result_list:
                result_list.append(n)
    compute_list = [i for i in compute_list if i not in all_gather_dep_list]
    result_list.extend(compute_list)

    return result_list


def bucket_reduce_scatter_by_block(
    sched: "scheduler.Scheduler",
    snodes: List["scheduler.BaseSchedulerNode"],
) -> List["scheduler.BaseSchedulerNode"]:
    """
    Bucket REDUCE_SCATTER and RS_WAIT by block
    """
    inverse_users, node_users = compute_node_users(snodes)

    # get the block each node belongs to
    node_block_list = []
    last_module = ""
    for node in snodes:
        if isinstance(node, scheduler.FusedSchedulerNode) or isinstance(
            node, scheduler.GroupedSchedulerNode
        ):
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
    rs_wait_list = []
    rs_wait_dep_list = []
    last_module = node_block_list[0]

    for idx, node in enumerate(snodes):
        current_module = node_block_list[idx]
        if current_module != last_module and len(reduce_scatter_list) > 0:
            # bucketing in the block boundary
            assert len(reduce_scatter_list) == len(rs_wait_list)
            (merged_reduce_scatter, rs_ir_node, copy_in_size) = merge_reducescatter(
                sched, reduce_scatter_list
            )
            rs_wait_dep_list = [r for r in rs_wait_dep_list if r not in result_list]
            merged_wait = merge_rs_wait(
                sched,
                rs_wait_list,
                reduce_scatter_list,
                rs_ir_node,
                copy_in_size,
                rs_wait_dep_list,
            )

            for n in [merged_reduce_scatter, merged_wait]:
                if n not in result_list:
                    result_list.append(n)

            reduce_scatter_list = []
            rs_wait_list = []
            rs_wait_dep_list = []

        if get_node_type(node) == NodeType.REDUCE_SCATTER:
            # add the small reduce_scatter to bucket
            reduce_scatter_list.append(node)
        elif (
            get_node_type(node) == NodeType.RS_WAIT
            and "reduce_scatter_tensor" in node.node.inputs[0].python_kernel_name
        ):
            # add the small rs_wait to bucket
            rs_wait_list.append(node)
            rs_wait_dep_list.extend(list(node_users[node]))
        else:
            # we do not bucket other nodes
            if node not in result_list and node not in rs_wait_dep_list:
                result_list.append(node)
        last_module = current_module
    assert len(reduce_scatter_list) == len(rs_wait_list)

    if len(reduce_scatter_list) > 0:
        (merged_reduce_scatter, rs_ir_node, copy_in_size) = merge_reducescatter(
            sched, reduce_scatter_list
        )
        rs_wait_dep_list = [r for r in rs_wait_dep_list if r not in result_list]
        merged_wait = merge_rs_wait(
            sched,
            rs_wait_list,
            reduce_scatter_list,
            rs_ir_node,
            copy_in_size,
            rs_wait_dep_list,
        )
        for n in [merged_reduce_scatter, merged_wait]:
            if n not in result_list:
                result_list.append(n)

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


def get_block_level(node: "scheduler.BaseSchedulerNode") -> str:
    """
    Get the node's block name
    """
    node_origin_list = []
    node_origin_list += node.node.origins
    module_list = []
    pattern = r"_modules\['([^']+)'\]"
    for n in node_origin_list:
        module_stack = n.meta.get("nn_module_stack", {})
        module_list_meta = list(module_stack.values())
        current_module_list = [""]
        while len(module_list_meta) > 1:
            module_info, block_info = module_list_meta.pop(0)
            module_info = re.findall(pattern, module_info)
            module_info = ".".join(module_info)
            current_module_list.append(module_info)
            if "layers." in module_info:
                break
        if current_module_list[-1] != "":
            module_list.append(current_module_list[-1])
    if len(module_list) > 0:
        module_list.sort()
        return max(module_list, key=module_list.count)
    return ""
