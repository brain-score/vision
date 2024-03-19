# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

def normalize(tensor, mean, std):
    """
    Args:
        tensor (Tensor): Tensor to normalize

    Returns:
        Tensor: Normalized tensor
    """
    tensor.sub_(mean).div_(std)
    return tensor
