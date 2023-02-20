import torch as ch
try:
    import dill
except: # Hack to work with brainscore environment
    import pickle
    dill=pickle
import os
from .tools import helpers, constants
from .attacker import AttackerModel

class FeatureExtractor(ch.nn.Module):
    '''
    Tool for extracting layers from models.

    Args:
        submod (torch.nn.Module): model to extract activations from
        layers (list of functions): list of functions where each function,
            when applied to submod, returns a desired layer. For example, one
            function could be `lambda model: model.layer1`.

    Returns:
        A model whose forward function returns the activations from the layers
            corresponding to the functions in `layers` (in the order that the
            functions were passed in the list).
    '''
    def __init__(self, submod, layers):
        # layers must be in order
        super(FeatureExtractor, self).__init__()
        self.submod = submod
        self.layers = layers
        self.n = 0

        for layer_func in layers:
            layer = layer_func(self.submod)
            def hook(module, _, output):
                module.register_buffer('activations', output)

            layer.register_forward_hook(hook)

    def forward(self, *args, **kwargs):
        """
        """
        # self.layer_outputs = {}
        out = self.submod(*args, **kwargs)
        activs = [layer_fn(self.submod).activations for layer_fn in self.layers]
        return [out] + activs

def make_and_restore_model(*_, arch, dataset, resume_path=None,
         parallel=True, pytorch_pretrained=False, strict=True,
         remap_checkpoint_keys={}, append_name_front_keys=None,
         arch_kwargs={}):
    """
    Makes a model and (optionally) restores it from a checkpoint.

    Args:
        arch (str|nn.Module): Model architecture identifier or otherwise a
            torch.nn.Module instance with the classifier
        dataset (Dataset class [see datasets.py])
        resume_path (str): optional path to checkpoint
        parallel (bool): if True, wrap the model in a DataParallel 
            (default True, recommended)
        pytorch_pretrained (bool): if True, try to load a standard-trained 
            checkpoint from the torchvision library (throw error if failed)
        strict (bool): If true, the state dict must exactly match, if False
            loading ignores non-matching keys
        remap_checkpoint_keys (dict): Modifies keys in the loaded state_dict 
            to new names, so that we can load old models if the code has changed. 
        append_name_front_keys (list): if not none, for each element of the list 
            makes new keys in the state dict that have the element appended to the front
            of the name. Useful for transfer models, if they were saved without the attacker model class. 
    Returns: 
        A tuple consisting of the model (possibly loaded with checkpoint), and the checkpoint itself
    """
    classifier_model = dataset.get_model(arch, pytorch_pretrained, arch_kwargs) if \
                            isinstance(arch, str) else arch

    model = AttackerModel(classifier_model, dataset)

    # optionally resume from a checkpoint
    checkpoint = None
    if resume_path:
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            if ch.cuda.is_available():
                checkpoint = ch.load(resume_path, pickle_module=dill)
            else:
                checkpoint = ch.load(resume_path, pickle_module=dill, map_location=ch.device('cpu'))
            
            # Makes us able to load models saved with legacy versions
            state_dict_path = 'model'
            if not ('model' in checkpoint):
                state_dict_path = 'state_dict'

            sd = checkpoint[state_dict_path]
            if append_name_front_keys is not None:
                new_sd = {}
                for key_idx in range(len(append_name_front_keys)):
                    sd_temp = {'%s%s'%(append_name_front_keys[key_idx], k):v for k,v in sd.items()}
                    new_sd.update(sd_temp)
                sd = new_sd

            sd = {k[len('module.'):]:v for k,v in sd.items()}
            # Load models if the keys changed slightly
            for old_key, new_key in remap_checkpoint_keys.items():
                print('mapping %s to %s'%(old_key, new_key))
                if type(new_key) is list: # If there are multiple keys that should be the same value (ie with attacker model)
                    for new_key_temp in new_key:
                        sd[new_key_temp] = sd[old_key]
                    del sd[old_key]
                else:
                    sd[new_key]=sd.pop(old_key)
            model.load_state_dict(sd, strict=strict)
            if parallel:
                model = ch.nn.DataParallel(model)
            if ch.cuda.is_available():           
                model = model.cuda()
 
            print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint.get('epoch', 'epoch number not found')))
        else:
            error_msg = "=> no checkpoint found at '{}'".format(resume_path)
            raise ValueError(error_msg)

    return model, checkpoint

def model_dataset_from_store(s, overwrite_params={}, which='last'):
    '''
    Given a store directory corresponding to a trained model, return the
    original model, dataset object, and args corresponding to the arguments.
    '''
    # which options: {'best', 'last', integer}
    if type(s) is tuple:
        s, e = s
        s = cox.store.Store(s, e, mode='r')

    m = s['metadata']
    df = s['metadata'].df

    args = df.to_dict()
    args = {k:v[0] for k,v in args.items()}
    fns = [lambda x: m.get_object(x), lambda x: m.get_pickle(x)]
    conds = [lambda x: m.schema[x] == s.OBJECT, lambda x: m.schema[x] == s.PICKLE]
    for fn, cond in zip(fns, conds):
        args = {k:(fn(v) if cond(k) else v) for k,v in args.items()}

    args.update(overwrite_params)
    args = Parameters(args)

    data_path = os.path.expandvars(args.data)
    if not data_path:
        data_path = '/tmp/'

    dataset = DATASETS[args.dataset](data_path)

    if which == 'last':
        resume = os.path.join(s.path, constants.CKPT_NAME)
    elif which == 'best':
        resume = os.path.join(s.path, constants.CKPT_NAME_BEST)
    else:
        assert isinstance(which, int), "'which' must be one of {'best', 'last', int}"
        resume = os.path.join(s.path, ckpt_at_epoch(which))

    model, _ = make_and_restore_model(arch=args.arch, dataset=dataset,
                                      resume_path=resume, parallel=False)
    return model, dataset, args
