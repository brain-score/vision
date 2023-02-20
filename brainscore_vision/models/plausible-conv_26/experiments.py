import torch
import torch.nn as nn
import torchvision
import numpy as np
from networks import network_utils, resnet, mobilenetv3, cornet_s
import utils
import datasets
import distributed_utils
import subprocess
import os


def make_cornet_s_imagenet(args, device='cpu'):
    net = cornet_s.CORnet_S(args.cornet_scale)
    if args.is_locally_connected:
        network_utils.convert_network_to_locally_connected(net, args)
    net = net.to(device)
    return net


def make_resnet18_imagenet(args, device='cpu'):
    net = resnet.resnet18(width_per_group=args.resnet18_width_per_group)
    if args.is_locally_connected:
        network_utils.convert_network_to_locally_connected(net, args)
    net = net.to(device)
    return net


def make_resnet_cifar(args, device='cpu'):
    net = resnet.resnet_6n2_cifar(n=args.resnet_cifar_n,
                                  num_classes=network_utils.get_task_dimensions(args.dataset)[-1])
    if args.is_locally_connected:
        network_utils.convert_network_to_locally_connected(net, args)
    net = net.to(device)
    return net


def make_mobilenetv3_imagenet(args, device='cpu'):
    net = torchvision.models.mobilenet_v3_small()
    if args.is_locally_connected:
        network_utils.convert_network_to_locally_connected(net, args)

    net = net.to(device)
    print(net)
    return net


def make_net(args, device):
    # if len(args.load_network) > 0:
    #     if torch.cuda.device_count() > 1:
    #         return torch.load(args.load_network, map_location={'cuda:0': device})
    #     return torch.load(args.load_network).to(device)

    if 'resnet18_imagenet' in args.experiment:
        net = make_resnet18_imagenet(args, device)
    elif args.experiment == 'mobilenetv3_imagenet':
        return make_mobilenetv3_imagenet(args, device)
    elif args.experiment == 'resnet_cifar':
        net = make_resnet_cifar(args, device)
    elif args.experiment == 'cornet_s_imagenet':
        net = make_cornet_s_imagenet(args, device)
    else:
        raise NotImplementedError('experiment must be mobilenetv3_imagenet or resnet18_imagenet, '
                                  'but %s was given' % args.experiment)
    network_utils.traverse_module(net, lambda x: network_utils.freeze_layers(x, args))

    print(net)
    return net


def run_locally_connected_experiment(net, args, device, world_size=1):
    """
    Runs the experiment with a vgg-like network.
    :param args:    Namespace from utils.parse_arguments
    :param device:  torch.device
    :return: np.array of float, np.array of float, np.array of float; train_acc, val_acc, test_acc (in percentage)
    """
    if isinstance(device, int):
        rank = device
        device = 'cuda:%d' % rank
    else:
        if world_size > 1:
            raise TypeError('For world_size > 1, device should be an int (=rank), but %d was given' % device)
        rank = None

    validation_ratio, validation_size, record_train_acc, record_val_acc, record_test_acc = \
        utils.configure_training_mode(args)

    if len(args.imagenet_path) == 0:
        imagenet_path = None
    else:
        imagenet_path = args.imagenet_path

    if args.dataset == 'TinyImageNet':
        if not os.path.exists('/tmp/romanp/tiny-imagenet-200'):
            subprocess.run(['mkdir -p /tmp/romanp/'], shell=True)
            subprocess.run(['cp -r /ceph/scratch/romanp/tiny-imagenet-200 /tmp/romanp/'], shell=True)
            print('\n\n\nMOVED TINYIMAGENET TO GPU\n\n\n')

    train_loader, validation_loader, test_loader = datasets.build_loaders_by_dataset(
        args.dataset, args.batch_size, args.image_padding, args.n_repetitions,
        validation_ratio=validation_ratio, train_validation_split_seed=0, do_cifar10_flip=not args.no_cifar10_flip,
        rank=rank, world_size=world_size, input_scale=args.input_scale, validation_size=validation_size,
        imagenet_path=imagenet_path)

    optimizer_class, optimizer_arguments_dict, lr_scheduler_arguments_dict = \
        utils.build_optimizer_and_parameters(args.optimizer, args.lr, args.weight_decay,
                                             args.sgd_momentum, args.epoch_decrease_lr, args.opt_lr_decrease_rate)

    loss = nn.CrossEntropyLoss()
    optimizer = optimizer_class(net.parameters(), **optimizer_arguments_dict)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **lr_scheduler_arguments_dict)

    weight_sharing_frequency = args.weight_sharing_frequency
    print('\n\n\nweight_sharing_frequency=%d\n\n\n' % weight_sharing_frequency)

    train_acc, val_acc, test_acc, locally_connected_weight_snr = utils.train_network(
        net, device, loss, optimizer, lr_scheduler, args.n_epochs,
        train_loader, validation_loader, test_loader,
        record_train_acc=record_train_acc, record_val_acc=record_val_acc, record_test_acc=record_test_acc,
        print_results=True, stat_report_frequency=args.stat_report_frequency,
        weight_sharing_training=args.share_weights, weight_sharing_frequency=weight_sharing_frequency,
        instant_weight_sharing=args.instant_weight_sharing, with_amp=args.amp,
        checkpoint_frequency=args.checkpoint_frequency,
        checkpoint_path=args.experiment_folder + ('/checkpoints/%s.pt' % args.job_idx))

    # todo: make this an argument, add a instant/not argument, add that to utils, add that to grid search

    if args.save_results:
        if world_size > 1:
            np.save(args.results_filename + ('_train_acc_%s' % device), train_acc)
            np.save(args.results_filename + ('_val_acc_%s' % device), val_acc)
            np.save(args.results_filename + ('_test_acc_%s' % device), test_acc)
            np.save(args.results_filename + ('locally_connected_weight_snr_%s' % device), locally_connected_weight_snr)
        else:
            np.save(args.results_filename + '_train_acc', train_acc)
            np.save(args.results_filename + '_val_acc', val_acc)
            np.save(args.results_filename + '_test_acc', test_acc)
            np.save(args.results_filename + 'locally_connected_weight_snr', locally_connected_weight_snr)

    if np.all(val_acc == -1):
        final_val_acc = 0.0
        print('\nLearning failed, all val_acc are -1!!!\n')
    else:
        final_val_acc = val_acc[val_acc[:, 0] != -1, 0][-1]
    print('Final validation accuracy: %.2f\non device %s' % (final_val_acc, device))


def main():
    args = utils.parse_arguments()
    device = args.device #torch.device(args.device if torch.cuda.is_available() else "cpu")

    if ('cuda' in device or 'gpu' in device) and not torch.cuda.is_available():
        raise ValueError('No cuda available')

    print(device)

    with np.printoptions(precision=4, suppress=True):
        net = make_net(args, device)
        run_locally_connected_experiment(net, args, device)


def ddp_main(rank, world_size):
    args = utils.parse_arguments()

    if len(args.load_network) == 0:
        raise NotImplementedError('Currently the only correct way is to pre-save network from grid search')

    distributed_utils.setup(rank, world_size)

    with np.printoptions(precision=4, suppress=True):
        net = make_net(args, 'cuda:%d' % rank)
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        ddp_net = distributed_utils.DDP(net, device_ids=[rank])
        torch.distributed.barrier()

        run_locally_connected_experiment(ddp_net, args, rank, world_size)
        # if len(args.save_final_network) > 0 and rank == 0:
        #     torch.save(net, args.save_final_network)

    distributed_utils.cleanup()


if __name__ == '__main__':
    if torch.cuda.device_count() > 1:
        raise NotImplementedError('Checkpoints, weight sharing and final results are not finalized for DDP!')
        distributed_utils.run_ddp(ddp_main, torch.cuda.device_count())
    else:
        main()
