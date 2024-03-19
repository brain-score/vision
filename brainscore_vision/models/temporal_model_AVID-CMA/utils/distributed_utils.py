# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import distributed as dist


def _gather_from_all(tensor):
    """
    Gather tensors from all gpus
    """
    gathered_tensor = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_tensor, tensor)
    gathered_tensor = torch.cat(gathered_tensor, 0)
    return gathered_tensor