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

def bucketing_per_blcok(
    snodes: List["scheduler.BaseSchedulerNode"],
) -> List["scheduler.BaseSchedulerNode"]:
        
    return snodes
