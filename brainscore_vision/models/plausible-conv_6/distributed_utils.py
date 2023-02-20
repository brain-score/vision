import os
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)  # gloo backend is slower?


def cleanup():
    dist.destroy_process_group()


def run_ddp(ddp_main, world_size):
    mp.spawn(ddp_main,
             args=(world_size,),
             nprocs=world_size,
             join=True)


class DDP(DistributedDataParallel):
    # Keeps the attributes of the underlying class
    # https://github.com/pytorch/pytorch/issues/16885
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
