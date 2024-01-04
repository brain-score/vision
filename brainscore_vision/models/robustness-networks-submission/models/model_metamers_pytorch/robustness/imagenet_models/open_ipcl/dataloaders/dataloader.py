import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter 
from torch.utils.data._utils.collate import default_collate_err_msg_format, np_str_obj_array_pattern
from PIL import Image
from pdb import set_trace

try:
    from torch._six import container_abcs, string_classes, int_classes
except:
    # some issue with torch v1.9
    import collections.abc as container_abcs 
    from torch._six import string_classes
    int_classes = int
    
try:
    from fastcore.foundation import delegates
except:
    from fastcore.meta import delegates
    
@delegates()
class FastLoader(DataLoader):
    def __init__(self, dataset, collate_fn=None, after_batch=None, **kwargs):
        collate_fn = FastLoader.default_collate if collate_fn is None else collate_fn
        super().__init__(dataset, collate_fn=collate_fn, **kwargs)
        self.after_batch = after_batch
        
    def __iter__(self):
        if self.num_workers == 0:
            _iter = _SingleProcessDataLoaderIter(self)
        else:
            _iter = _MultiProcessingDataLoaderIter(self)
        
        for batch in _iter:
            if self.after_batch: batch = self.after_batch(batch)
            yield batch
            
    @staticmethod    
    def default_collate(batch, to_float=True):
        r"""Puts each data field into a tensor with outer dimension batch size
        
            A typical batch will be a list of lists. e.g.,
                [[TensorImages], [TensorLongLabels]]
            
            Each of those sub lists will be passed separately to this default_collate
            function, and handled based on type.
            
            ...a list of tensors will be stacked. 
            ...a list of PIL images will be converted to np.array, then stacked in a tensor
        """
        
        elem = batch[0]
        elem_type = type(elem)
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.stack(batch, 0, out=out)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':            
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                    raise TypeError(default_collate_err_msg_format.format(elem.dtype))
                
                return FastLoader.default_collate([torch.as_tensor(b) for b in batch])
            elif elem.shape == ():  # scalars
                return torch.as_tensor(batch)
        elif isinstance(elem, Image.Image):
            h,w,c = np.array(elem).shape
            tensor = torch.zeros( (len(batch), c, h, w), dtype=torch.uint8 )

            for i, img in enumerate(batch):
                nump_array = np.asarray(img, dtype=np.uint8)
                if(nump_array.ndim < 3):
                    nump_array = np.expand_dims(nump_array, axis=-1)
                nump_array = np.moveaxis(nump_array, -1, 0)
                tensor[i] += torch.from_numpy(nump_array)
            
            if to_float: tensor = tensor.float()
                
            return tensor
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: FastLoader.default_collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
            return elem_type(*(FastLoader.default_collate(samples) for samples in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):                
            transposed = zip(*batch)
            return [FastLoader.default_collate(samples) for samples in transposed]

        raise TypeError(default_collate_err_msg_format.format(elem_type))
      