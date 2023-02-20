import torch as ch
import numpy as np
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torchvision.utils import make_grid
from cox.utils import Parameters

from .tools import helpers
from .tools.helpers import AverageMeter, calc_fadein_eps, \
        save_checkpoint, ckpt_at_epoch, has_attr
from .tools import constants as consts
import dill 
import time

from .tools.warm_up_scheduler import GradualWarmupScheduler

import os
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm as tqdm

def check_required_args(args, eval_only=False):
    """
    Check that the required training arguments are present.

    Args:
        args (argparse object): the arguments to check
        eval_only (bool) : whether to check only the arguments for evaluation
    """
    required_args_eval = ["adv_eval"]
    required_args_train = ["epochs", "out_dir", "adv_train",
        "log_iters", "lr", "momentum", "weight_decay"]
    adv_required_args = ["attack_steps", "eps", "constraint", "use_best",
                        "eps_fadein_epochs", "attack_lr", "random_restarts"]

    # Generic function for checking all arguments in a list
    def check_args(args_list):
        for arg in args_list:
            assert has_attr(args, arg), f"Missing argument {arg}"

    # Different required args based on training or eval:
    if not eval_only: check_args(required_args_train)
    else: check_args(required_args_eval)
    # More required args if we are robustly training or evaling
    is_adv = bool(args.adv_train) or bool(args.adv_eval)
    if is_adv:
        check_args(adv_required_args)
    # More required args if the user provides a custom training loss
    has_custom_train = has_attr(args, 'custom_train_loss')
    has_custom_adv = has_attr(args, 'custom_adv_loss')
    if has_custom_train and is_adv and not has_custom_adv:
        raise ValueError("Cannot use custom train loss \
            without a custom adversarial loss (see docs)")

def make_optimizer_and_schedule(args, model, checkpoint, params):
    """
    *Internal Function* (called directly from train_model)

    Creates an optimizer and a schedule for a given model, restoring from a
    checkpoint if it is non-null.

    Args:
        args (object) : an arguments object, see
            :meth:`~robustness.train.train_model` for details
        model (AttackerModel) : the model to create the optimizer for
        checkpoint (dict) : a loaded checkpoint saved by this library and loaded
            with `ch.load`
        params (list|None) : a list of parameters that should be updatable, all
            other params will not update. If ``None``, update all params 

    Returns:
        An optimizer (ch.nn.optim.Optimizer) and a scheduler
            (ch.nn.optim.lr_schedulers module).
    """
    # Make optimizer
    param_list = model.parameters() if params is None else params
    optimizer = SGD(param_list, args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # Make schedule
    schedule = None
    if args.step_lr:
        schedule = lr_scheduler.StepLR(optimizer, step_size=args.step_lr)
    elif args.custom_schedule == 'cyclic':
        eps = args.epochs
        lr_func = lambda t: np.interp([t], [0, eps*2//5, eps], [0, args.lr, 0])[0]
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)
    elif args.custom_schedule:
        cs = args.custom_schedule
        periods = eval(cs) if type(cs) is str else cs
        def lr_func(ep):
            for (milestone, lr) in reversed(periods):
                if ep > milestone: return lr/args.lr
            return args.lr
        schedule = lr_scheduler.LambdaLR(optimizer, lr_func)

    if args.warm_up_lr:   
        schedule = GradualWarmupScheduler(optimizer, multiplier=1, 
                                          total_epoch=1, 
                                          after_scheduler=schedule)

    # Fast-forward the optimizer and the scheduler if resuming
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        try:
            schedule.load_state_dict(checkpoint['schedule'])
        except:
            steps_to_take = checkpoint['epoch']
            print('Could not load schedule (was probably LambdaLR).'
                  f' Stepping {steps_to_take} times instead...')
            for i in range(steps_to_take):
                schedule.step()

    return optimizer, schedule

def eval_model(args, model, loader, store, adv_only=False):
    """
    Evaluate a model for standard (and optionally adversarial) accuracy.

    Args:
        args (object) : A list of arguments---should be a python object 
            implementing ``getattr()`` and ``setattr()``.
        model (AttackerModel) : model to evaluate
        loader (iterable) : a dataloader serving `(input, label)` batches from
            the validation set
        store (cox.Store) : store for saving results in (via tensorboardX)
        adv_only (bool) : set to True to run only the adversarial evaluation, default False
    """
    check_required_args(args, eval_only=True)
    start_time = time.time()

    if store is not None: 
        store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA)
    writer = store.tensorboard if store else None

    model = ch.nn.DataParallel(model)

    if not adv_only:
        prec1, nat_loss = _model_loop(args, 'val', loader, 
                                      model, None, 0, False, writer, store)
    else:
        prec1, nat_loss = float('nan'), float('nan')

    adv_prec1, adv_loss = float('nan'), float('nan')
    if args.adv_eval: 
        args.eps = eval(str(args.eps)) if has_attr(args, 'eps') else None
        args.attack_lr = eval(str(args.attack_lr)) if has_attr(args, 'attack_lr') else None
        adv_prec1, adv_loss = _model_loop(args, 'val', loader, 
                                        model, None, 0, True, writer, store)
    log_info = {
        'epoch':0,
        'nat_prec1':prec1,
        'adv_prec1':adv_prec1,
        'nat_loss':nat_loss,
        'adv_loss':adv_loss,
        'train_prec1':float('nan'),
        'train_loss':float('nan'),
        'time': time.time() - start_time
    }

    # Log info into the logs table
    if store: store[consts.LOGS_TABLE].append_row(log_info)
    return log_info

def train_model(args, model, loaders, *, checkpoint=None, 
                store=None, update_params=None):
    """
    Main function for training a model. 

    Args:
        args (object) : A python object for arguments, implementing
            ``getattr()`` and ``setattr()`` and having the following
            attributes. See :attr:`robustness.defaults.TRAINING_ARGS` for a 
            list of arguments, and you can use
            :meth:`robustness.defaults.check_and_fill_args` to make sure that
            all required arguments are filled and to fill missing args with
            reasonable defaults:

            adv_train (int or bool, *required*)
                if 1/True, adversarially train, otherwise if 0/False do 
                standard training
            epochs (int, *required*)
                number of epochs to train for
            lr (float, *required*)
                learning rate for SGD optimizer
            weight_decay (float, *required*)
                weight decay for SGD optimizer
            momentum (float, *required*)
                momentum parameter for SGD optimizer
            step_lr (int)
                if given, drop learning rate by 10x every `step_lr` steps
            custom_schedule (str)
                If given, use a custom LR schedule (format: [(epoch, LR),...])
            warm_up_lr (bool)
                If given, warm up the learning rate in the first epoch. 
            adv_eval (int or bool)
                If True/1, then also do adversarial evaluation, otherwise skip
                (ignored if adv_train is True)
            log_iters (int, *required*)
                How frequently (in epochs) to save training logs
            save_ckpt_iters (int, *required*)
                How frequently (in epochs) to save checkpoints (if -1, then only
                save latest and best ckpts)
            attack_lr (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack step size
            constraint (str, *required if adv_train or adv_eval*)
                the type of adversary constraint
                (:attr:`robustness.attacker.STEPS`)
            eps (float or str, *required if adv_train or adv_eval*)
                float (or float-parseable string) for the adv attack budget
            attack_steps (int, *required if adv_train or adv_eval*)
                number of steps to take in adv attack
            eps_fadein_epochs (int, *required if adv_train or adv_eval*)
                If greater than 0, fade in epsilon along this many epochs
            use_best (int or bool, *required if adv_train or adv_eval*) :
                If True/1, use the best (in terms of loss) PGD step as the
                attack, if False/0 use the last step
            random_restarts (int, *required if adv_train or adv_eval*)
                Number of random restarts to use for adversarial evaluation
            custom_train_loss (function, optional)
                If given, a custom loss instead of the default CrossEntropyLoss.
                Takes in `(logits, targets)` and returns a scalar.
            custom_adv_loss (function, *required if custom_train_loss*)
                If given, a custom loss function for the adversary. The custom
                loss function takes in `model, input, target` and should return
                a vector representing the loss for each element of the batch, as
                well as the classifier output.
            regularizer (function, optional) 
                If given, this function of `model, input, target` returns a
                (scalar) that is added on to the training loss without being
                subject to adversarial attack
            clip_grad_op (function, optional)
                If given, this function runs on model.parameters() to clip the 
                gradients between each training iteration. Can be helpful for 
                exploding gradients. 
            iteration_hook (function, optional)
                If given, this function is called every training iteration by
                the training loop (useful for custom logging). The function is
                given arguments `model, iteration #, loop_type [train/eval],
                current_batch_ims, current_batch_labels`.
            epoch hook (function, optional)
                Similar to iteration_hook but called every epoch instead, and
                given arguments `model, log_info` where `log_info` is a
                dictionary with keys `epoch, nat_prec1, adv_prec1, nat_loss,
                adv_loss, train_prec1, train_loss`.

        model (AttackerModel) : the model to train.
        loaders (tuple[iterable]) : `tuple` of data loaders of the form
            `(train_loader, val_loader)` 
        checkpoint (dict) : a loaded checkpoint previously saved by this library
            (if resuming from checkpoint)
        store (cox.Store) : a cox store for logging training progress
        train_params (list) : list of parameters to use for training, if None
            then all parameters in the model are used (useful for transfer
            learning)
    """
    # Logging setup
    writer = store.tensorboard if store else None
    if store is not None: 
        # TODO: Set up the schema so that it makes entries for each of the multitask elements
        if args.custom_train_loss is not None:
            if (type(args.custom_train_loss.all_loss_weights)==dict):
                store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA_MULTITASK)
            else:
                store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA)
        else:
            store.add_table(consts.LOGS_TABLE, consts.LOGS_SCHEMA)
        store.add_table(consts.CKPTS_TABLE, consts.CKPTS_SCHEMA)
    
    # Reformat and read arguments
    check_required_args(args) # Argument sanity check
    args.eps = eval(str(args.eps)) if has_attr(args, 'eps') else None
    args.attack_lr = eval(str(args.attack_lr)) if has_attr(args, 'attack_lr') else None

    # Initial setup
    train_loader, val_loader = loaders
    opt, schedule = make_optimizer_and_schedule(args, model, checkpoint, update_params)

    best_prec1, start_epoch = (0, 0)
    if checkpoint:
        print('LOADING EPOCH FOR OPTIMIZER SCHEDULE')
        print(checkpoint['epoch'])
        start_epoch = checkpoint['epoch']
        try:
            best_prec1 = checkpoint[f"{'adv' if args.adv_train else 'nat'}_prec1"]
        except:
            print('Missing prec in checkpoint')

    # Put the model into parallel mode
    assert not hasattr(model, "module"), "model is already in DataParallel."
    model = ch.nn.DataParallel(model).cuda()

    # Timestamp for training start time
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        train_prec1, train_loss = _model_loop(args, 'train', train_loader, 
                model, opt, epoch, args.adv_train, writer, store)
        last_epoch = (epoch == (args.epochs - 1))

        # evaluate on validation set
        sd_info = {
            'model':model.state_dict(),
            'optimizer':opt.state_dict(),
            'schedule':(schedule and schedule.state_dict()),
            'epoch': epoch+1
        }

        def save_checkpoint(filename):
            ckpt_save_path = os.path.join(args.out_dir if not store else \
                                          store.path, filename)
            ch.save(sd_info, ckpt_save_path, pickle_module=dill)

        save_checkpoint('temp_checkpoint')

        save_its = args.save_ckpt_iters
        should_save_ckpt = (epoch % save_its == 0) and (save_its > 0)
        should_log = (epoch % args.log_iters == 0)

        if should_log or last_epoch or should_save_ckpt:
            # log + get best
            with ch.no_grad():
                prec1, nat_loss = _model_loop(args, 'val', val_loader, model, 
                        None, epoch, False, writer, store)

            # loader, model, epoch, input_adv_exs
            should_adv_eval = args.adv_eval or args.adv_train
            adv_val = should_adv_eval and _model_loop(args, 'val', val_loader,
                    model, None, epoch, True, writer, store)
            adv_prec1, adv_loss = adv_val or (-1.0, -1.0)

            # remember best prec@1 and save checkpoint
            our_prec1 = adv_prec1 if args.adv_train else prec1
            if isinstance(our_prec1, dict):
                print('Best Criteria Not Specified for Multi Task Models')
                is_best = True # Always set the "best" to be the latest
                best_prec1 = our_prec1
            else:
                is_best = our_prec1 > best_prec1
                best_prec1 = max(our_prec1, best_prec1)

            # Convert multitask readouts to strings so we save all values
            # TODO: make some way to set up the SCHEMA to be dependent on the keys
            if isinstance(prec1, dict) or isinstance(adv_prec1, dict):
                if isinstance(prec1, dict):
                    prec1 = '|'.join(['%s:%s'%(p_key, p_value) for p_key, p_value in prec1.items()])
                if isinstance(adv_prec1, dict):
                    adv_prec1 = '|'.join(['%s:%s'%(p_key, p_value) for p_key, p_value in adv_prec1.items()])
                if isinstance(prec1, float):
                    prec1 = str(prec1)
                if isinstance(adv_prec1, float):
                    adv_prec1 = str(adv_prec1)
                train_prec1 = '|'.join(['%s:%s'%(p_key, p_value) for p_key, p_value in train_prec1.items()])

#             if isinstance(prec1, dict) or isinstance(adv_prec1, dict):
#                 # log every checkpoint
#                 log_info = {
#                     'epoch':epoch + 1,
#                     'nat_loss':nat_loss,
#                     'adv_loss':adv_loss,
#                     'train_loss':train_loss,
#                     'time': time.time() - start_time
#                 }
#                 if isinstance(prec1, dict):
#                     for prec_key in prec1.keys():
#                         log_info['nat_prec1_%s'%prec_key] = prec1[prec_key]
#                         log_info['train_prec1_%s'%prec_key] = train_prec1[prec_key]
#                 else:
#                     log_info['nat_prec1']=prec1
#                 if isinstance(adv_prec1, dict):
#                     for prec_key in adv_prec1.keys():
#                         log_info['adv_prec1_%s'%prec_key] = adv_prec1[prec_key]
#                         log_info['train_prec1_%s'%prec_key] = train_prec1[prec_key]
#                 else:
#                     log_info['adv_prec1']=adv_prec1
#                         
#             else:
            # log every checkpoint
            log_info = {
                'epoch':epoch + 1,
                'nat_prec1':prec1,
                'adv_prec1':adv_prec1,
                'nat_loss':nat_loss,
                'adv_loss':adv_loss,
                'train_prec1':train_prec1,
                'train_loss':train_loss,
                'time': time.time() - start_time
            }

            # Log info into the logs table
            if store: store[consts.LOGS_TABLE].append_row(log_info)
            # If we are at a saving epoch (or the last epoch), save a checkpoint
            if should_save_ckpt or last_epoch: save_checkpoint(ckpt_at_epoch(epoch))
            # If  store exists and this is the last epoch, save a checkpoint
            if last_epoch and store: store[consts.CKPTS_TABLE].append_row(sd_info)

            # Update the latest and best checkpoints (overrides old one)
            save_checkpoint(consts.CKPT_NAME_LATEST)
            if is_best: save_checkpoint(consts.CKPT_NAME_BEST)

        if schedule: schedule.step()
        if has_attr(args, 'epoch_hook'): args.epoch_hook(model, log_info)

    return model

def _model_loop(args, loop_type, loader, model, opt, epoch, adv, writer, store):
    """
    *Internal function* (refer to the train_model and eval_model functions for
    how to train and evaluate models).

    Runs a single epoch of either training or evaluating.

    Args:
        args (object) : an arguments object (see
            :meth:`~robustness.train.train_model` for list of arguments
        loop_type ('train' or 'val') : whether we are training or evaluating
        loader (iterable) : an iterable loader of the form 
            `(image_batch, label_batch)`
        model (AttackerModel) : model to train/evaluate
        opt (ch.optim.Optimizer) : optimizer to use (ignored for evaluation)
        epoch (int) : which epoch we are currently on
        adv (bool) : whether to evaluate adversarially (otherwise standard)
        writer : tensorboardX writer (optional)
        store (cox.Store) : store for saving results in (via tensorboardX)

    Returns:
        The average top1 accuracy and the average loss across the epoch.
    """
    if not loop_type in ['train', 'val']:
        err_msg = "loop_type ({0}) must be 'train' or 'val'".format(loop_type)
        raise ValueError(err_msg)
    is_train = (loop_type == 'train')

    # Check if the model has multiple losses. If it does then use a multi=task printout
    try:
        multi_task_loss = (type(args.custom_train_loss.all_loss_weights)==dict)
        multi_task_keys = args.custom_train_loss.all_loss_weights.keys()
    except AttributeError:
        multi_task_loss = False
   
    losses = AverageMeter()
    if multi_task_loss:
        top1 = {task_key: AverageMeter() for task_key \
                    in args.custom_train_loss.all_loss_weights.keys()}
        top5 = {task_key: AverageMeter() for task_key \
                    in args.custom_train_loss.all_loss_weights.keys()}
    else:
        top1 = AverageMeter()
        top5 = AverageMeter()

    prec = 'NatPrec' if not adv else 'AdvPrec'
    loop_msg = 'Train' if loop_type == 'train' else 'Val'

    # switch to train/eval mode depending
    model = model.train() if is_train else model.eval()

    # If adv training (or evaling), set eps and random_restarts appropriately
    if adv:
        eps = calc_fadein_eps(epoch, args.eps_fadein_epochs, args.eps) \
                if is_train else args.eps
        random_restarts = 0 if is_train else args.random_restarts

    # Custom training criterion
    has_custom_train_loss = has_attr(args, 'custom_train_loss')
    train_criterion = args.custom_train_loss if has_custom_train_loss \
            else ch.nn.CrossEntropyLoss()

    if has_attr(train_criterion, 'loss_type_name'):
        if train_criterion.loss_type_name == 'CTC':
            pass
    else:
        pass
    
    has_custom_adv_loss = has_attr(args, 'custom_adv_loss')
    adv_criterion = args.custom_adv_loss if has_custom_adv_loss else None

    attack_kwargs = {}
    if adv:
        attack_kwargs = {
            'constraint': args.constraint,
            'eps': eps,
            'step_size': args.attack_lr,
            'iterations': args.attack_steps,
            'random_start': args.random_start,
            'custom_loss': adv_criterion,
            'random_restarts': random_restarts,
            'use_best': bool(args.use_best)
        }

    def save_checkpoint(filename):
        ckpt_save_path = os.path.join(args.out_dir if not store else \
                                      store.path, filename)
        ch.save(sd_info, ckpt_save_path, pickle_module=dill)

    iterator = tqdm(enumerate(loader), total=len(loader), dynamic_ncols=False)

    for i, (inp, target) in iterator:
        if i % 2000 == 0: # Save a temporary checkpoint every 2000 steps
            if is_train:
                # Make some stuff to save temporary checkpoints during training
                sd_info = {
                    'model':model.state_dict(),
                    'optimizer':opt.state_dict(),
                    # 'schedule':(schedule and schedule.state_dict()),
                    'epoch': epoch,
                }
                sd_info['iteration'] = i
                save_checkpoint('checkpoint_epoch%d_iter%d.ckpt'%(epoch,i))
   
       # measure data loading time
        try:
            target = target.cuda(non_blocking=True)
        except(AttributeError): # Catch a dictionary target
            for key in list(target.keys()):
                target[key] = target[key].cuda(non_blocking=True)
          
        output, final_inp = model(inp, target=target, make_adv=adv,
                                  **attack_kwargs)
        loss = train_criterion(output, target)

        if len(loss.shape) > 0: loss = loss.mean()

        if isinstance(output, tuple):
            model_logits = output[0]
            print_target = target
        elif isinstance(output, dict):
            model_logits = {}
            print_target = {}
            for output_key, output_value in output.items():
                model_logits[output_key] = output_value
                print_target[output_key] = target[output_key]
#             print_key = 'signal/word_int' # TODO: make this flexible to work with other tasks
#             model_logits = output[print_key]
#             print_target = target[print_key]
        else:
            model_logits = output
            print_target=target
        
        # measure accuracy and record loss
        top1_acc = float('nan')
        top5_acc = float('nan')
        losses.update(loss.item(), inp.size(0))

        if isinstance(model_logits, dict):
            # TODO print out the individual loss for each of the tasks
            top1_acc = {}
            top5_acc = {}
            for task_key in multi_task_keys:
                top1_acc[task_key] = float('nan')
                top5_acc[task_key] = float('nan')
                if len(print_target[task_key].shape) == 1:
                    maxk = min(5, model_logits[task_key].shape[-1])
                    prec1, prec5 = helpers.accuracy(model_logits[task_key], print_target[task_key], topk=(1, maxk))
                    top1[task_key].update(prec1[0], inp.size(0))
                    top5[task_key].update(prec5[0], inp.size(0)) 
                    top1_acc[task_key] = top1[task_key].avg
                    top5_acc[task_key] = top5[task_key].avg
                else:
                    top1_acc
        
        else:
            maxk = min(5, model_logits.shape[-1])
            prec1, prec5 = helpers.accuracy(model_logits, print_target, topk=(1, maxk))

            top1.update(prec1[0], inp.size(0))
            top5.update(prec5[0], inp.size(0))

            top1_acc = top1.avg
            top5_acc = top5.avg

        reg_term = 0.0
        if has_attr(args, "regularizer"):
            reg_term =  args.regularizer(model, inp, target)
        loss = loss + reg_term

        # compute gradient and do SGD step
        if is_train:
            opt.zero_grad()
            loss.backward()
            if has_attr(args, "clip_grad_op"): 
                args.clip_grad_op(model.parameters(), **args.clip_grad_kwargs)
            # Warm up lr for the first few steps
            if has_attr(args, "warm_up_lr") and (epoch==0) and (i<500):
                # In the first 500 steps of training warm up the learning rate
                for g in opt.param_groups:               
                    new_lr = g['initial_lr'] / (500-i)
                    g['lr'] = new_lr
            opt.step()
        elif adv and i == 0 and writer:
            # add some examples to the tensorboard
            nat_grid = make_grid(inp[:15, ...])
            adv_grid = make_grid(final_inp[:15, ...])
            if len(inp.shape)==4:
                try:
                    writer.add_image('Nat input', nat_grid, epoch)
                    writer.add_image('Adv input', adv_grid, epoch)
                except:
                    print('Adv not type that can be written to tensorboard')
            elif len(inp.shape)==3:
                print('TODO: make writer for audio')

        # ITERATOR
        if isinstance(top1_acc, dict):
            task_string = ['{task_key} | {0}1 {top1_acc:.3f} | '
                            '{0}5 {top5_acc:.3f} |'.format(prec, top1_acc=top1_acc[task_key],
                            top5_acc=top5_acc[task_key], task_key=task_key) for task_key in top1_acc.keys()]
            desc = ('{2} Epoch:{0} | Loss {loss.avg:.4f} | '
                    '{1} | Reg term: {reg} |'.format(epoch, ''.join(task_string),
                    loop_msg, loss=losses, reg=reg_term))
        else:
            desc = ('{2} Epoch:{0} | Loss {loss.avg:.4f} | '
                    '{1}1 {top1_acc:.3f} | {1}5 {top5_acc:.3f} | '
                    'Reg term: {reg} ||'.format( epoch, prec, loop_msg, 
                    loss=losses, top1_acc=top1_acc, top5_acc=top5_acc, reg=reg_term))

        # USER-DEFINED HOOK
        if has_attr(args, 'iteration_hook'):
            args.iteration_hook(model, i, loop_type, inp, target)

        iterator.set_description(desc)
        iterator.refresh()

    if writer is not None:
        prec_type = 'adv' if adv else 'nat'
        descs = ['loss', 'top1', 'top5']
        vals = [losses, top1, top5]
        for d, v in zip(descs, vals):
            if isinstance(v, dict):
                for v_key, v_value in v.items():
                    writer.add_scalar('_'.join([prec_type, loop_type, d, v_key]), 
                                      v_value.avg, epoch)
            else:
                writer.add_scalar('_'.join([prec_type, loop_type, d]), v.avg,
                                  epoch)

    if isinstance(top1, dict):
        return {top1_key:top1_value.avg for top1_key, top1_value in top1.items()}, losses.avg
    else:
        return top1.avg, losses.avg

