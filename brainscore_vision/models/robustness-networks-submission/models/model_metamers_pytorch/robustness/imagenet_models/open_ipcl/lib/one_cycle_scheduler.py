from numbers import Number
from typing import Any, AnyStr, Callable, Collection, Dict, Hashable, Iterator, List, Mapping, NewType, Optional
from typing import Sequence, Tuple, TypeVar, Union
from types import SimpleNamespace
from abc import abstractmethod, abstractproperty
from collections import abc,  Counter, defaultdict, Iterable, namedtuple, OrderedDict
from IPython.core.debugger import set_trace
from torch import nn, optim, as_tensor
import numpy as np

def is_listy(x:Any)->bool: return isinstance(x, (tuple,list))
def is_tuple(x:Any)->bool: return isinstance(x, tuple)
def is_dict(x:Any)->bool: return isinstance(x, dict)
def is_pathlike(x:Any)->bool: return isinstance(x, (str,Path))
def noop(x): return x

AnnealFunc = Callable[[Number,Number,float], Number]
ListOrItem = Union[Collection[Any],int,float,str]
OptListOrItem = Optional[ListOrItem]
StartOptEnd=Union[float,Tuple[float,float]]
Floats = Union[float, Collection[float]]
OptOptimizer = Optional[optim.Optimizer]

def annealing_linear(start:Number, end:Number, pct:float)->Number:
    "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start + pct * (end-start)
def annealing_exp(start:Number, end:Number, pct:float)->Number:
    "Exponentially anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    return start * (end/start) ** pct
def annealing_cos(start:Number, end:Number, pct:float)->Number:
    "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
    cos_out = np.cos(np.pi * pct) + 1
    return end + (start-end)/2 * cos_out

def ifnone(a:Any,b:Any)->Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a

def listify(p:OptListOrItem=None, q:OptListOrItem=None):
    "Make `p` listy and the same length as `q`."
    if p is None: p=[]
    elif isinstance(p, str):          p = [p]
    elif not isinstance(p, Iterable): p = [p]
    #Rank 0 tensors in PyTorch are Iterable but don't have a length.
    else:
        try: a = len(p)
        except: p = [p]
    n = q if type(q)==int else len(p) if q is None else len(q)
    if len(p)==1: p = p * n
    assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)

class Callback(object):
    "Base class for callbacks that want to record values, dynamically change learner params, etc."
    _order=0
    def on_train_begin(self, **kwargs:Any)->None:
        "To initialize constants in the callback."
        pass
    def on_epoch_begin(self, **kwargs:Any)->None:
        "At the beginning of each epoch."
        pass
    def on_batch_begin(self, **kwargs:Any)->None:
        "Set HP before the output and loss are computed."
        pass
    def on_loss_begin(self, **kwargs:Any)->None:
        "Called after forward pass but before loss has been computed."
        pass
    def on_backward_begin(self, **kwargs:Any)->None:
        "Called after the forward pass and the loss has been computed, but before backprop."
        pass
    def on_backward_end(self, **kwargs:Any)->None:
        "Called after backprop but before optimizer step. Useful for true weight decay in AdamW."
        pass
    def on_step_end(self, **kwargs:Any)->None:
        "Called after the step of the optimizer but before the gradients are zeroed."
        pass
    def on_batch_end(self, **kwargs:Any)->None:
        "Called at the end of the batch."
        pass
    def on_epoch_end(self, **kwargs:Any)->None:
        "Called at the end of an epoch."
        pass
    def on_train_end(self, **kwargs:Any)->None:
        "Useful for cleaning up things and saving files/models."
        pass
    def jump_to_epoch(self, epoch)->None:
        "To resume training at `epoch` directly."
        pass

    def get_state(self, minimal:bool=True):
        "Return the inner state of the `Callback`, `minimal` or not."
        to_remove = ['exclude', 'not_min'] + getattr(self, 'exclude', []).copy()
        if minimal: to_remove += getattr(self, 'not_min', []).copy()
        return {k:v for k,v in self.__dict__.items() if k not in to_remove}

    def  __repr__(self):
        attrs = func_args(self.__init__)
        to_remove = getattr(self, 'exclude', [])
        list_repr = [self.__class__.__name__] + [f'{k}: {getattr(self, k)}' for k in attrs if k != 'self' and k not in to_remove]
        return '\n'.join(list_repr)

class Scheduler():
    "Used to \"step\" from start,end (`vals`) over `n_iter` iterations on a schedule defined by `func`"
    def __init__(self, vals:StartOptEnd, n_iter:int, func:Optional[AnnealFunc]=None):
        self.start,self.end = (vals[0],vals[1]) if is_tuple(vals) else (vals,0)
        self.n_iter = max(1,n_iter)
        if func is None: self.func = annealing_linear if is_tuple(vals) else annealing_no
        else:          self.func = func
        self.n = 0
        # set_trace()
        
    def restart(self): self.n = 0

    def step(self)->Number:
        "Return next value along annealed schedule."
        self.n += 1
        return self.func(self.start, self.end, self.n/self.n_iter)

    @property
    def is_done(self)->bool:
        "Return `True` if schedule completed."
        return self.n >= self.n_iter

    
class OneCycleScheduler(Callback):
    "Manage 1-Cycle style training as outlined in Leslie Smith's [paper](https://arxiv.org/pdf/1803.09820.pdf)."
    def __init__(self, lr_max:float, num_batches:int, opt:OptOptimizer, moms:Floats=(0.95,0.85), div_factor:float=25., pct_start:float=0.3,
                 final_div:float=None, tot_epochs:int=None, start_epoch:int=None):
        # super().__init__(learn)
        self.lr_max,self.num_batches,self.opt,self.div_factor,self.pct_start,self.final_div = lr_max,num_batches,opt,div_factor,pct_start,final_div
        if self.final_div is None: self.final_div = div_factor*1e4
        self.moms=tuple(listify(moms,2))
        if is_listy(self.lr_max): self.lr_max = np.array(self.lr_max)
        self.start_epoch, self.tot_epochs = start_epoch, tot_epochs
        
    def steps(self, *steps_cfg:StartOptEnd):
        "Build anneal schedule for all of the parameters."
        return [Scheduler(step, n_iter, func=func)
                for (step,(n_iter,func)) in zip(steps_cfg, self.phases)]

    def on_train_begin(self, n_epochs:int, epoch:int, **kwargs:Any)->None:
        "Initialize our optimization params based on our annealing schedule."        
        res = {'epoch':self.start_epoch} if self.start_epoch is not None else None
        self.start_epoch = ifnone(self.start_epoch, epoch)
        self.tot_epochs = ifnone(self.tot_epochs, n_epochs)        
        n = self.num_batches * self.tot_epochs
        a1 = int(n * self.pct_start)
        a2 = n-a1
        self.phases = ((a1, annealing_cos), (a2, annealing_cos))
        low_lr = self.lr_max/self.div_factor
        self.lr_scheds = self.steps((low_lr, self.lr_max), (self.lr_max, self.lr_max/self.final_div))
        self.mom_scheds = self.steps(self.moms, (self.moms[1], self.moms[0]))
        # self.opt = self.learn.opt
        
        lr, mom = self.lr_scheds[0].start, self.mom_scheds[0].start
        for param_group in self.opt.param_groups: 
            param_group['lr'] = lr
            # param_group['momentum'] = mom
            if 'momentum' in param_group: param_group['momentum'] = mom
            if 'betas' in param_group: param_group['betas'] = (mom, param_group['betas'][1])
        # self.opt.lr,self.opt.mom = self.lr_scheds[0].start,self.mom_scheds[0].start
        
        self.idx_s = 0
        # set_trace()
        return res
    
    def jump_to_epoch(self, epoch:int)->None:
        for _ in range(self.num_batches * epoch):
            self.on_batch_end(True)

    def on_batch_end(self, train, **kwargs:Any)->None:
        "Take one step forward on the annealing schedule for the optim params."
        if train:                
            if self.idx_s >= len(self.lr_scheds): return {'stop_training': True, 'stop_epoch': True}
            # self.opt.lr = self.lr_scheds[self.idx_s].step()
            # self.opt.mom = self.mom_scheds[self.idx_s].step()
            lr, mom = self.lr_scheds[self.idx_s].step(), self.mom_scheds[self.idx_s].step()
            for param_group in self.opt.param_groups: 
                param_group['lr'] = lr
                if 'momentum' in param_group: param_group['momentum'] = mom
                if 'betas' in param_group: param_group['betas'] = (mom, param_group['betas'][1])
            # when the current schedule is complete we move onto the next
            # schedule. (in 1-cycle there are two schedules)
            if self.lr_scheds[self.idx_s].is_done:                
                self.idx_s += 1

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Tell Learner to stop if the cycle is finished."
        if epoch > self.tot_epochs: return {'stop_training': True}

class FlatToAnnealingLR(Callback):
    "Linearly rise from max_lr/div_factor to max_lr, then annealing_cos to final_lr."
    def __init__(self, lr_max:float, num_batches:int, opt:OptOptimizer, div_factor:float=25., pct_start:float=0.3,
                 final_div:float=None, tot_epochs:int=None, start_epoch:int=None):
        self.lr_max,self.num_batches,self.opt,self.div_factor,self.pct_start,self.final_div = lr_max,num_batches,opt,div_factor,pct_start,final_div
        if self.final_div is None: self.final_div = div_factor*1e4
        if is_listy(self.lr_max): self.lr_max = np.array(self.lr_max)
        self.start_epoch, self.tot_epochs = start_epoch, tot_epochs
        
    def steps(self, *steps_cfg:StartOptEnd):
        "Build anneal schedule for all of the parameters."
        return [Scheduler(step, n_iter, func=func)
                for (step,(n_iter,func)) in zip(steps_cfg, self.phases)]

    def on_train_begin(self, n_epochs:int, epoch:int, **kwargs:Any)->None:
        "Initialize our optimization params based on our annealing schedule."        
        res = {'epoch':self.start_epoch} if self.start_epoch is not None else None
        self.start_epoch = ifnone(self.start_epoch, epoch)
        self.tot_epochs = ifnone(self.tot_epochs, n_epochs)        
        n = self.num_batches * self.tot_epochs
        a1 = int(n * self.pct_start)
        a2 = n-a1
        self.phases = ((a1, annealing_linear), (a2, annealing_cos))
        low_lr = self.lr_max/self.div_factor
        self.lr_scheds = self.steps((low_lr, self.lr_max), (self.lr_max, self.lr_max/self.final_div))
        # self.opt = self.learn.opt
        
        lr = self.lr_scheds[0].start
        for param_group in self.opt.param_groups: 
            param_group['lr'] = lr
        
        self.idx_s = 0

        return res
    
    def jump_to_epoch(self, epoch:int)->None:
        for _ in range(self.num_batches * epoch):
            self.on_batch_end(True)

    def on_batch_end(self, train, **kwargs:Any)->None:
        "Take one step forward on the annealing schedule for the optim params."
        if train:                
            if self.idx_s >= len(self.lr_scheds): return {'stop_training': True, 'stop_epoch': True}
            
            lr = self.lr_scheds[self.idx_s].step()
            for param_group in self.opt.param_groups: 
                param_group['lr'] = lr
                
            # when the current schedule is complete we move onto the next
            # schedule. (in 1-cycle there are two schedules)
            if self.lr_scheds[self.idx_s].is_done:                
                self.idx_s += 1

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Tell Learner to stop if the cycle is finished."
        if epoch > self.tot_epochs: return {'stop_training': True}
        
class OneCycleSchedulerTau(Callback):
    def __init__(self, lemniscate:Any, tau:Tuple, num_batches:int, div_factor:float=25., pct_start:float=0.3, final_div:float=None, tot_epochs:int=None, start_epoch:int=None):
    
        self.lemniscate,self.tau,self.num_batches,self.div_factor,self.pct_start,self.final_div = lemniscate,tau,num_batches,div_factor,pct_start,final_div
        #if self.final_div is None: self.final_div = div_factor*1e4
        #if is_listy(self.tau_max): self.tau_max = np.array(self.tau_max)
        self.start_epoch, self.tot_epochs = start_epoch, tot_epochs

    def steps(self, *steps_cfg:StartOptEnd):
        "Build anneal schedule for all of the parameters."
        return [Scheduler(step, n_iter, func=func)
                for (step,(n_iter,func)) in zip(steps_cfg, self.phases)]
    
    def on_train_begin(self, n_epochs:int, epoch:int, **kwargs:Any)->None:
        "Initialize our optimization params based on our annealing schedule."        
        res = {'epoch':self.start_epoch} if self.start_epoch is not None else None
        self.start_epoch = ifnone(self.start_epoch, epoch)
        self.tot_epochs = ifnone(self.tot_epochs, n_epochs)        
        n = self.num_batches * self.tot_epochs
        a1 = int(n * self.pct_start)
        a2 = n-a1
        self.phases = ((a1, annealing_cos), (a2, annealing_cos))
        # low_lr = self.tau_max/self.div_factor
        # self._scheds = self.steps((low_lr, self.tau_max), (self.tau_max, self.tau_max/self.final_div))
        self._scheds = self.steps((self.tau[0], self.tau[1]), (self.tau[1], self.tau[0]))
        
        val = self._scheds[0].start
        self.lemniscate.params[1] = val
        
        self.idx_s = 0
        
        return res
    
    def jump_to_epoch(self, epoch:int)->None:
        for _ in range(self.num_batches * epoch):
            self.on_batch_end(True)

    def on_batch_end(self, train, **kwargs:Any)->None:
        "Take one step forward on the annealing schedule for the optim params."
        if train:                
            if self.idx_s >= len(self._scheds): return {'stop_training': True, 'stop_epoch': True}
            
            val = self._scheds[self.idx_s].step()
            self.lemniscate.params[1] = val

            # when the current schedule is complete we move onto the next
            # schedule. (in 1-cycle there are two schedules)
            if self._scheds[self.idx_s].is_done:                
                self.idx_s += 1

    def on_epoch_end(self, epoch, **kwargs:Any)->None:
        "Tell Learner to stop if the cycle is finished."
        if epoch > self.tot_epochs: return {'stop_training': True}


def show_tau_schedule(tau_scheduler, num_epochs, num_batches):
    epochs = []
    tau = []
    tau_scheduler.on_train_begin(epoch=0, n_epochs=num_epochs)
    mb = master_bar(range(num_epochs))
    for epoch in mb:
        ts = []
        for batch_no in progress_bar(range(num_batches), parent=mb):
            ts.append(tau_scheduler.lemniscate.params[1].cpu().numpy().tolist())
            tau_scheduler.on_batch_end(True)
        epochs.append(epoch)
        tau.append(np.mean(ts))
        
    ax = sns.lineplot(x="epoch", y="tau", data={"epoch":epochs ,"tau": tau})

    return epochs, tau, ax        