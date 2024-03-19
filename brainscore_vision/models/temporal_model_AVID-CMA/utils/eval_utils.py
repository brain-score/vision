# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn
import torch.distributed as dist

from . import logger
from . import main_utils
import yaml
import os


def prepare_environment(args, cfg, fold):
    if args.distributed:
        while True:
            try:
                dist.init_process_group(backend='nccl', init_method='tcp://localhost:{}'.format(args.port), world_size=args.world_size, rank=args.gpu)
                break
            except RuntimeError:
                args.port = str(int(args.port) + 1)

    model_cfg = yaml.safe_load(open(args.model_cfg))['model']
    eval_dir = '{}/{}/eval-{}/fold-{:02d}'.format(model_cfg['model_dir'], model_cfg['name'], cfg['benchmark']['name'], fold)
    os.makedirs(eval_dir, exist_ok=True)
    yaml.safe_dump(cfg, open('{}/config.yaml'.format(eval_dir), 'w'))

    logger = utils.logger.Logger(quiet=args.quiet, log_fn='{}/eval.log'.format(eval_dir), rank=args.gpu)
    if any(['SLURM' in env for env in list(os.environ.keys())]):
        logger.add_line("=" * 30 + "   SLURM   " + "=" * 30)
        for env in os.environ.keys():
            if 'SLURM' in env:
                logger.add_line('{:30}: {}'.format(env, os.environ[env]))
    logger.add_line("=" * 30 + "   Config   " + "=" * 30)
    def print_dict(d, ident=''):
        for k in d:
            if isinstance(d[k], dict):
                logger.add_line("{}{}".format(ident, k))
                print_dict(d[k], ident='  '+ident)
            else:
                logger.add_line("{}{}: {}".format(ident, k, str(d[k])))
    print_dict(cfg)
    logger.add_line("=" * 30 + "   Model Config   " + "=" * 30)
    print_dict(model_cfg)

    return eval_dir, model_cfg, logger


def distribute_model_to_cuda(model, args, cfg):
    if torch.cuda.device_count() == 1:
        model = model.cuda()
    elif args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        cfg['dataset']['batch_size'] = max(cfg['dataset']['batch_size'] // args.world_size, 1)
        cfg['num_workers'] = max(cfg['num_workers'] // args.world_size, 1)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        model = torch.nn.DataParallel(model).cuda()
    return model


def build_dataloader(db_cfg, split_cfg, fold, num_workers, distributed):
    import torch.utils.data as data
    from datasets import preprocessing
    if db_cfg['transform'] == 'msc+color':
        video_transform = preprocessing.VideoPrep_MSC_CJ(
            crop=(db_cfg['crop_size'], db_cfg['crop_size']),
            num_frames=int(db_cfg['video_fps'] * db_cfg['clip_duration']),
            pad_missing=True,
            augment=split_cfg['use_augmentation'],
            min_area=db_cfg['min_area'],
            color=db_cfg['color'],
        )
    elif db_cfg['transform'] == 'crop+color':
        video_transform = preprocessing.VideoPrep_Crop_CJ(
            crop=(db_cfg['crop_size'], db_cfg['crop_size']),
            num_frames=int(db_cfg['video_fps'] * db_cfg['clip_duration']),
            pad_missing=True,
            augment=split_cfg['use_augmentation'],
        )
    else:
        raise ValueError

    import datasets
    if db_cfg['name'] == 'ucf101':
        dataset = datasets.UCF
    elif db_cfg['name'] == 'hmdb51':
        dataset = datasets.HMDB
    elif db_cfg['name'] == 'kinetics':
        dataset = datasets.Kinetics
    else:
        raise ValueError('Unknown dataset')

    db = dataset(
        subset=split_cfg['split'].format(fold=fold),
        return_video=True,
        video_clip_duration=db_cfg['clip_duration'],
        video_fps=db_cfg['video_fps'],
        video_transform=video_transform,
        return_audio=False,
        return_labels=True,
        mode=split_cfg['mode'],
        clips_per_video=split_cfg['clips_per_video'],
    )

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(db)
    else:
        sampler = None

    drop_last = split_cfg['drop_last'] if 'drop_last' in split_cfg else True
    loader = data.DataLoader(
        db,
        batch_size=db_cfg['batch_size']  if split_cfg['mode'] == 'clip' else max(1, db_cfg['batch_size']//split_cfg['clips_per_video']),
        num_workers=num_workers,
        pin_memory=True,
        shuffle=(sampler is None) and split_cfg['use_shuffle'],
        sampler=sampler,
        drop_last=drop_last
    )
    return loader


def build_dataloaders(cfg, fold, num_workers, distributed, logger):
    logger.add_line("=" * 30 + "   Train DB   " + "=" * 30)
    train_loader = build_dataloader(cfg, cfg['train'], fold, num_workers, distributed)
    logger.add_line(str(train_loader.dataset))

    logger.add_line("=" * 30 + "   Test DB   " + "=" * 30)
    test_loader = build_dataloader(cfg, cfg['test'], fold, num_workers, distributed)
    logger.add_line(str(test_loader.dataset))

    logger.add_line("=" * 30 + "   Dense DB   " + "=" * 30)
    dense_loader = build_dataloader(cfg, cfg['test_dense'], fold, num_workers, distributed)
    logger.add_line(str(dense_loader.dataset))

    return train_loader, test_loader, dense_loader


class CheckpointManager(object):
    def __init__(self, checkpoint_dir, rank=0):
        self.checkpoint_dir = checkpoint_dir
        self.best_metric = 0.
        self.rank = rank

    def save(self, model, optimizer, scheduler, epoch, eval_metric=0.):
        if self.rank is not None and self.rank != 0:
            return
        is_best = False
        if eval_metric > self.best_metric:
            self.best_metric = eval_metric
            is_best = True

        main_utils.save_checkpoint(state={
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
        }, is_best=is_best, model_dir=self.checkpoint_dir)

    def last_checkpoint_fn(self):
        return '{}/checkpoint.pth.tar'.format(self.checkpoint_dir)

    def best_checkpoint_fn(self):
        return '{}/model_best.pth.tar'.format(self.checkpoint_dir)

    def checkpoint_fn(self, last=False, best=False):
        assert best or last
        assert not (last and best)
        if last:
            return self.last_checkpoint_fn()
        if best:
            return self.best_checkpoint_fn()

    def checkpoint_exists(self, last=False, best=False):
        return os.path.isfile(self.checkpoint_fn(last, best))

    def restore(self, model, optimizer, scheduler, restore_last=False, restore_best=False):
        checkpoint_fn = self.checkpoint_fn(restore_last, restore_best)
        ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})
        start_epoch = ckp['epoch']
        model.load_state_dict(ckp['state_dict'])
        optimizer.load_state_dict(ckp['optimizer'])
        scheduler.load_state_dict(ckp['scheduler'])
        return start_epoch


class ClassificationWrapper(torch.nn.Module):
    def __init__(self, feature_extractor, n_classes, feat_name, feat_dim, pooling_op=None, use_dropout=False, dropout=0.5):
        super(ClassificationWrapper, self).__init__()
        self.feature_extractor = feature_extractor
        self.feat_name = feat_name
        self.use_dropout = use_dropout
        if pooling_op is not None:
            self.pooling = eval('torch.nn.'+pooling_op)
        else:
            self.pooling = None
        if use_dropout:
            self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(feat_dim, n_classes)

    def forward(self, *inputs):
        emb = self.feature_extractor(*inputs, return_embs=True)[self.feat_name]
        emb_pool = self.pooling(emb) if self.pooling is not None else emb
        emb_pool = emb_pool.view(inputs[0].shape[0], -1)
        if self.use_dropout:
            emb_pool = self.dropout(emb_pool)
        logit = self.classifier(emb_pool)
        return logit


class Classifier(nn.Module):
    def __init__(self, n_classes, feat_name, feat_dim, pooling, l2_norm=False, use_bn=True, use_dropout=False):
        super(Classifier, self).__init__()
        self.use_bn = use_bn
        self.feat_name = feat_name
        self.pooling = eval('nn.'+pooling) if pooling is not None else None
        self.l2_norm = l2_norm
        if use_bn:
            self.bn = nn.BatchNorm1d(feat_dim)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout()
        self.classifier = nn.Linear(feat_dim, n_classes)

    def forward(self, x):
        with torch.no_grad():
            if self.use_dropout:
                x = self.dropout(x)
            if self.l2_norm:
                x = nn.functional.normalize(x, p=2, dim=-1)
            if self.pooling is not None and len(x.shape) > 2:
                x = self.pooling(x)
            x = x.view(x.shape[0], -1).contiguous().detach()
        if self.use_bn:
            x = self.bn(x)
        return self.classifier(x)


class MOSTCheckpointManager(object):
    def __init__(self, checkpoint_dir, rank=0):
        self.rank = rank
        self.checkpoint_dir = checkpoint_dir
        self.best_metric = 0.

    def save(self, model, optimizer, epoch, eval_metric=0.):
        if self.rank != 0:
            return
        is_best = False
        if eval_metric > self.best_metric:
            self.best_metric = eval_metric
            is_best = True

        try:
            state_dict =  model.classifiers.state_dict()
        except AttributeError:
            state_dict = model.module.classifiers.state_dict()
        main_utils.save_checkpoint(state={
            'epoch': epoch,
            'state_dict': state_dict,
            'optimizer': optimizer.state_dict(),
        }, is_best=is_best, model_dir=self.checkpoint_dir)

    def last_checkpoint_fn(self):
        return '{}/checkpoint.pth.tar'.format(self.checkpoint_dir)

    def best_checkpoint_fn(self):
        return '{}/model_best.pth.tar'.format(self.checkpoint_dir)

    def checkpoint_fn(self, last=False, best=False):
        assert best or last
        # assert not (last and best)
        if last:
            return self.last_checkpoint_fn()
        elif best:
            return self.best_checkpoint_fn()

    def checkpoint_exists(self, last=False, best=False):
        return os.path.isfile(self.checkpoint_fn(last, best))

    def restore(self, model, optimizer, restore_last=False, restore_best=False):
        checkpoint_fn = self.checkpoint_fn(restore_last, restore_best)
        ckp = torch.load(checkpoint_fn, map_location={'cuda:0': 'cpu'})
        start_epoch = ckp['epoch']
        try:
            model.classifiers.load_state_dict(ckp['state_dict'])
        except AttributeError:
            model.module.classifiers.load_state_dict(ckp['state_dict'])
        optimizer.load_state_dict(ckp['optimizer'])
        return start_epoch


class MOSTModel(nn.Module):
    def __init__(self, feature_extractor, n_classes, feat_names, feat_dims, pooling_ops, l2_norm=None, use_bn=False, use_dropout=False):
        super(MOSTModel, self).__init__()
        assert len(feat_dims) == len(pooling_ops) == len(feat_names)
        n_outputs = len(feat_dims)
        self.feat_names = feat_names
        self.feat_dims = feat_dims
        self.pooling_ops = pooling_ops
        if l2_norm is None:
            l2_norm = [False] * len(feat_names)
        if not isinstance(l2_norm, list):
            l2_norm = [l2_norm] * len(feat_names)
        self.l2_norm = l2_norm

        feature_extractor.train(False)
        self.feature_extractor = feature_extractor

        self.classifiers = nn.ModuleList([
            Classifier(n_classes, feat_name=feat_names[i], feat_dim=feat_dims[i], pooling=pooling_ops[i], l2_norm=l2_norm[i], use_bn=use_bn, use_dropout=use_dropout) for i in range(n_outputs)
        ])

        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def forward(self, *x):
        with torch.no_grad():
            embs = self.feature_extractor(*x, return_embs=self.feat_names)
            embs = {ft: embs[ft] for ft in self.feat_names}

        for classifier, ft in zip(self.classifiers, self.feat_names):
            embs[ft] = classifier(embs[ft])
        return embs


def build_model(feat_cfg, eval_cfg, eval_dir, args, logger):
    import models
    pretrained_net = models.__dict__[feat_cfg['arch']](**feat_cfg['args'])

    # Load from checkpoint
    checkpoint_fn = '{}/{}/checkpoint.pth.tar'.format(feat_cfg['model_dir'], feat_cfg['name'])
    ckp = torch.load(checkpoint_fn, map_location='cpu')
    pretrained_net.load_state_dict({k.replace('module.', ''): ckp['model'][k] for k in ckp['model']})

    # Wrap with linear-head classifiers
    if eval_cfg['model']['name'] == 'ClassificationWrapper':
        model = ClassificationWrapper(feature_extractor=pretrained_net.video_model, **eval_cfg['model']['args'])
        ckp_manager = CheckpointManager(eval_dir, rank=args.gpu)
    elif eval_cfg['model']['name'] == 'MOSTWrapper':
        model = MOSTModel(feature_extractor=pretrained_net.video_model, **eval_cfg['model']['args'])
        ckp_manager = MOSTCheckpointManager(eval_dir, rank=args.gpu)
    else:
        raise ValueError

    # Log model description
    logger.add_line("=" * 30 + "   Model   " + "=" * 30)
    logger.add_line(str(model))
    logger.add_line("=" * 30 + "   Parameters   " + "=" * 30)
    logger.add_line(main_utils.parameter_description(model))
    logger.add_line("=" * 30 + "   Pretrained model   " + "=" * 30)
    logger.add_line("File: {}\nEpoch: {}".format(checkpoint_fn, ckp['epoch']))

    # Distribute
    model = distribute_model_to_cuda(model, args, eval_cfg)

    return model, ckp_manager


class BatchWrapper:
    def __init__(self, model, batch_size):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, x):
        outs = []
        for i in range(0, x.shape[0], self.batch_size):
            outs += [self.model(x[i:i + self.batch_size])]
        return torch.cat(outs, 0)