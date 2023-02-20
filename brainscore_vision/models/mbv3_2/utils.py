import torch
import torch.optim as optim
import numpy as np
import argparse
from networks import network_utils, locally_connected_utils
from warnings import warn


# to parse bool variables https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments(args=None):
    """
    Parse the arguments.
    :param args: None or list of str (e.g. ['--device', 'cuda:0']). If None, parses command line arguments. .
    :return: Namespace
    """
    parser = argparse.ArgumentParser(description='Configure the run')

    # general parameters
    parser.add_argument("--job-idx", type=str, default='0',
                        help='index of the current job (integer for grid search entries or final); default: 0')
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device; default: cuda:0")
    parser.add_argument('--training-mode', type=str, default='validation',
                        help='Train with validation or test; default: validation')
    parser.add_argument('--record-train-accuracy', action='store_true', default=False,
                        help='Record accuracy on test data (with transformations); default: False')
    parser.add_argument('--validation-split', type=float, default=0.1, help='Validaiton/full train ratio; default: 0.1')
    parser.add_argument('--validation-size', type=int, default=10000,
                        help='Validaiton size for ImageFolder datasets; default: 10000')
    parser.add_argument('--experiment', type=str, default='locally_connected',
                        help='Experiment type: locally_connected or resnet_cifar or resnet18_imagenet;'
                             ' default: locally_connected')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='Dataset: CIFAR10, CIFAR100, ImageNet; default: CIFAR10')
    parser.add_argument("--n-epochs", type=int, default=100, help="Number of epochs; default: 100")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size; default: 128")
    parser.add_argument("--experiment-folder", type=str, default='./',
                        help='Path to the folder with the current experiment; default: ./')
    parser.add_argument('--save-results', action='store_true', default=False,
                        help='Save the accuracy and SNR; default: False')
    parser.add_argument('--results-filename', type=str, default='logs/tmp',
                        help='Filename primer to save thee accuracy and SNR '
                             '(saved as filename_train_acc.npy and so on; default: logs/tmp')
    parser.add_argument('--stat-report-frequency', type=int, default=1,
                        help='Record the running stats every n epochs; default: 1')
    parser.add_argument('--checkpoint-frequency', type=int, default=10,
                        help='Record the running stats every n epochs; default: 10')
    parser.add_argument('--amp', action='store_true', default=False, help='Enables automatic mixed precision')
    parser.add_argument('--imagenet-path', type=str, default='',
                        help='Path to ImageNet or TinyImageNet. '
                             'If empty the default paths will be used; ddefault: (empty)')
    parser.add_argument('--cornet-scale', type=int, default=4, help='CORnet-S size scaling; default: 4')

    # optimizers
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='Optimizer for backprop: Adam, AdamW or SGD; default: SGD')
    parser.add_argument("--sgd-momentum", type=float, default=0.9,
                        help="SGD momentum for backprop; default: 0.9")
    parser.add_argument("--epoch-decrease-lr", type=int, default=100, nargs="+",
                        help="List of epochs for lr decrease (e.g. --epoch-decrease-lr 100 200); default: [100]")
    parser.add_argument("--opt-lr-decrease-rate", type=float, default=0.25,
                        help='Learning rate multiplier when it decreases; default: 0.25')
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay for backprop; default: 1e-5")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="Learning rate for backprop; default: 0.005")

    # network parameters
    parser.add_argument('--n-first-conv', type=int, default=0,
                        help='How many convolutions in the beginning of the network; default: 0')
    parser.add_argument('--conv-1x1', action='store_true', default=False,
                        help='Makes 1x1 LC layers of an LC net convolutional (so far only for mobilenetv3);'
                             ' default: False')
    parser.add_argument('--freeze-1x1', action='store_true', default=False, help='Freeze all 1x1 conv/LC layers')
    parser.add_argument('--old-1x1', action='store_true', default=False, help='1x1x1x1 not converted to conv')

    # approx conv parameters
    parser.add_argument("--is-locally-connected", action="store_true", default=False,
                        help="Use approximate convolutions; default: False")
    parser.add_argument("--locally-connected-deviation-eps", type=float, default=0.0,
                        help='Std of Gaussian noise around convolutional initial conditions; default: 0.0')

    parser.add_argument('--image-padding', type=int, default=0, help='Padding for RandomCrop; default: 0')
    parser.add_argument('--no-cifar10-flip', action='store_true', default=False,
                        help='Disable horizontal flips for CIFAR10; default: False')
    parser.add_argument('--n-repetitions', type=int, default=1,
                        help='Number of samples of each image within batch; default: 1')
    parser.add_argument('--locally-connected-inverse-ch-ch-connection-ratio', type=int, default=None,
                        help='Inverse fraction of how many (random) channels each layer sees;'
                             ' default: None (all channels)')
    parser.add_argument('--resnet18-width-per-group', type=int, default=64, help='Base width for resnet18; default: 64')
    parser.add_argument('--resnet-cifar-n', type=int, default=3,
                        help='Number of block within each ressolution for cifar resnets'
                             ' (6n+2 total layers for n blocks); default: 3')
    parser.add_argument('--input-scale', type=int, default=1, help='Scaling factor for images; default: 1')
    parser.add_argument("--dynamic-1x1", action="store_true", default=False,
                        help="Use approximate convolutions; default: False")
    parser.add_argument("--dynamic-NxN", action="store_true", default=False,
                        help="Use approximate convolutions; default: False")
    parser.add_argument("--dynamic-sharing-hebb-lr", type=float, default=0.1,
                        help="Learning rate for dynamic weight sharing in 1x1; default: 0.1")
    parser.add_argument('--dynamic-sharing-b-freq', type=int, default=8, help='Frequency of b generation; default: 8')

    # weight sharing
    parser.add_argument("--share-weights", action="store_true", default=False,
                        help="Enables weight sharing")
    parser.add_argument("--instant-weight-sharing", action="store_true", default=False,
                        help="Enables weight sharing without actual learning")
    parser.add_argument('--weight-sharing-frequency', type=int, default=1e3,
                        help='How often the weights are shared; default: 1e3')

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    if isinstance(args.epoch_decrease_lr, int):
        args.epoch_decrease_lr = [args.epoch_decrease_lr]

    if args.freeze_1x1 and not (args.conv_1x1 and args.is_locally_connected):
        raise NotImplementedError('--freeze_1x1 requires --conv_1x1 and --is_locally_connected')

    print("Simulation arguments:", args)

    return args


def configure_training_mode(args):
    """
    Configures test/validation runs.
    :param args: Namespace from parse_arguments
    :return: float, bool, bool, bool; validation_split, record_train_acc, record_val_acc, record_test_acc
    """
    if args.training_mode != 'test' and args.training_mode != 'validation':
        raise NotImplementedError('training-mode must be test or validation, but %s was given' % args.training_mode)

    validation_size = args.validation_size

    if args.training_mode == 'test':
        args.validation_split = 0.0
        validation_size = 0

    if args.validation_split == 0.0 or args.validation_size == 0:
        validation_size = 0
        warn('Validation ratio or validation_size are 0.0,'
             ' the validation set is the training one w/o random transformations')

    record_train_acc = args.record_train_accuracy
    record_val_acc = True
    record_test_acc = (args.training_mode == 'test')

    return args.validation_split, validation_size, record_train_acc, record_val_acc, record_test_acc


def compute_accuracy(output, target, topk=(1,)):
    # https://github.com/pytorch/examples/blob/447974f6337543d4de6b888e244a964d3c9b71f6/imagenet/main.py
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_one_epoch(net, device, data_loader, train_optimizer, net_loss_function,
                    weight_sharing_training, weight_sharing_frequency, grad_scaler, with_amp, instant_weight_sharing):
    net.train()

    total_loss = 0.0#, device=device)
    total_num = 0.0
    accuracy = np.zeros(2)#, device=device)

    for iteration, data in enumerate(data_loader):
        # BRING BACK FOR DATA AUG
        # images = torch.cat(tuple(single_input.to(device, non_blocking=True) for single_input in data[:-1]), dim=0)
        # labels = data[-1].to(device, non_blocking=True)
        # labels = torch.cat(tuple(labels for _ in range(len(data) - 1)), dim=0)

        images, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
        batch_size = images.shape[0]

        with torch.cuda.amp.autocast(enabled=with_amp):
            net_output = net(images)
            loss = net_loss_function(net_output, labels)

        total_loss += loss.item() * batch_size #  detach().clone() * batch_size  #

        train_optimizer.zero_grad()
        grad_scaler.scale(loss).backward()
        grad_scaler.step(train_optimizer)
        grad_scaler.update()

        with torch.no_grad():
            for module in net.modules():
                if isinstance(module, locally_connected_utils.LocallyConnected1x1HebbWeightSharing):
                    module.weights.data = module.weights.data - 1e-1 * (module.weights.data - module.exp_avg_weights)
                    module.exp_avg_weights = 0.9 * module.exp_avg_weights + 0.1 * module.weights.data
        # with torch.cuda.amp.autocast(enabled=False):
        #     with torch.no_grad():
        #         for module in net.modules():
        #             if isinstance(module, locally_connected_utils.LocallyConnected1x1HebbWeightSharing):
        #                 module.weights.data = module.weights.data - module.centered_lc.hebb_lr * (
        #                          torch.einsum('bgihwkc,bgjhw->igjhwkc',
        #                                       module.centered_lc.b.float(), module.centered_lc.output.float()).view(
        #                              module.weights.data.shape)
        #                          / module.centered_lc.b.shape[0]) - \
        #                                       1e-1 * (module.weights.data - module.exp_avg_weights)
        #                 module.exp_avg_weights = 0.9 * module.exp_avg_weights + 0.1 * module.weights.data
        #             if isinstance(module, locally_connected_utils.LocallyConnectedNxNHebbWeightSharing):
        #                 new_b = module.padding(
        #                                torch.kron(torch.ones(1, 1, int(np.ceil(module.in_size[0] / module.kernel_size)),
        #                                                      int(np.ceil(module.in_size[1] / module.kernel_size)),
        #                                                      device=module.b.device), module.b)[:, :, :module.in_size[0],
        #                                :module.in_size[1]]).unfold(
        #                     2, module.kernel_size, module.stride).unfold(3, module.kernel_size, module.stride).view(module.b.shape)
        #
        #                 module.weights.data = module.weights.data - module.hebb_lr * (
        #                          torch.einsum('bgihwkc,bgjhw->igjhwkc',
        #                                       new_b.float(), module.output.float()).view(
        #                              module.weights.data.shape)
        #                          / module.b.shape[0]) - \
        #                                       1e-1 * (module.weights.data - module.exp_avg_weights)
        #                 module.exp_avg_weights = 0.9 * module.exp_avg_weights + 0.1 * module.weights.data

        total_num += batch_size

        # accuracy_new = compute_accuracy(net_output, labels, (1, 5))
        # accuracy[0] += accuracy_new[0]
        # accuracy[1] += accuracy_new[1]
        with torch.no_grad():
            top5 = torch.topk(net_output, 5, dim=-1)[1]
            # todo: optimize with just one topk
            accuracy[0] += (top5[:, 0] == labels).sum().item()
            accuracy[1] += (top5 == labels[:, None]).sum().item()

        # todo: add amp here
        if weight_sharing_training and iteration % weight_sharing_frequency == 0:
            # print('weight sharing training')
            torch.cuda.empty_cache()

            for module in net.modules():
                if isinstance(module, locally_connected_utils.LocallyConnected2d):
                    if instant_weight_sharing:
                        module.share_weights_instantly()
                    else:
                        saved_weights = module.weights.detach().clone()
                        hebbian_lr = 1e-1
                        weight_gamma = 1e-5

                        for weight_sharing_iter in range(100):
                            images = torch.randn(128, module.in_channels, module.kernel_size, module.kernel_size,
                                                      device=images.device)
                            images = torch.kron(torch.ones(1, 1, int(np.ceil(module.in_size[0] / module.kernel_size)),
                                                      int(np.ceil(module.in_size[0] / module.kernel_size)),
                                                                device=images.device),
                                           images)[:, :, :module.in_size[0], :module.in_size[1]]
                            module.share_weights(images, hebbian_lr, weight_gamma, saved_weights)

        # print(iteration, flush=True)
        # print(network_utils.compute_locally_connected_non_padded_weight_average_snr(net), flush=True)

    print('Running training accuracy (top1, top5):\t %s on device %s' % (100 * accuracy / total_num,
                                                                         device))
    return total_loss / total_num


def test_one_epoch(net, device, data_loader):
    # todo: do I need amp here?
    net.eval()
    total = 0.0
    correct = torch.zeros(2, device=device)

    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            outputs = net(images)

            total += labels.size(0)
            correct[0] += (torch.topk(outputs.data, 1, dim=-1)[1] == labels[:, None]).sum()#.item()
            correct[1] += (torch.topk(outputs.data, 5, dim=-1)[1] == labels[:, None]).sum()#.item()

    return 100 * correct.cpu().numpy() / total


def record_and_print_running_statistics(record_train_acc, record_val_acc, record_test_acc, print_results,
                                        train_acc, val_acc, test_acc, epoch, net,
                                        train_loader, validation_loader, test_loader, record_locally_connected_weight_snr,
                                        locally_connected_weight_snr, device):
    """
    Records and prints running statistics.
    :param record_train_acc:    bool
    :param record_val_acc:      bool
    :param record_test_acc:     bool
    :param print_results:       bool, print accuracies if recorded
    :param train_acc:           np.array, train accuracy to write to
    :param val_acc:             np.array, validation accuracy to write to
    :param test_acc:            np.array, test accuracy to write to
    :param epoch:               int, epoch number
    :param net:                 nn.Module, network
    :param train_loader:        torch.utils.data.DataLoader
    :param validation_loader:   torch.utils.data.DataLoader
    :param test_loader:         torch.utils.data.DataLoader
    :param record_locally_connected_weight_snr: bool
    :param locally_connected_weight_snr: 2d np.array of size (n_epochs, n_conv_layers)
    :param device:              torch.device
    """
    if record_train_acc:
        train_acc[epoch] = test_one_epoch(net, device, train_loader)
        if print_results:
            print('Train accuracy (top1, top5):\t %s on device %s' % (train_acc[epoch], device))
    if record_val_acc:
        val_acc[epoch] = test_one_epoch(net, device, validation_loader)
        if print_results:
            print('Validation accuracy (top1, top5):\t %s on device %s' % (val_acc[epoch], device))
    if record_test_acc:
        test_acc[epoch] = test_one_epoch(net, device, test_loader)
        if print_results:
            print('Test accuracy (top1, top5):\t %s on device %s' % (test_acc[epoch], device))
    if record_locally_connected_weight_snr:
        locally_connected_weight_snr[epoch] = \
            network_utils.compute_locally_connected_non_padded_weight_average_snr(net)
        if print_results:
            print('Approx conv weight average SNR=|mean|/std:\n\t%s on device %s' %
                  (locally_connected_weight_snr[epoch], device))


def train_network(net, device, output_loss, optimizer,
                  scheduler, n_epochs, train_loader, validation_loader,
                  test_loader,  record_train_acc=False, record_val_acc=True,
                  record_test_acc=False, print_results=True, stat_report_frequency=1,
                  weight_sharing_training=False, weight_sharing_frequency=1e3, with_amp=False,
                  instant_weight_sharing=False, checkpoint_frequency=10, checkpoint_path='./checkpoint.pt'):
    """
    Trains the given network
    :param net:                 nn.Module, network
    :param device:              torch.device
    :param output_loss:         nn.Module, loss for the last layer
    :param optimizer:           torch.optim.Optimizer for the last layer or backprop
    :param scheduler:           torch.optim.lr_scheduler._LRScheduler,
        learning rate scheduler for the last layer or backprop
    :param n_epochs:            int, number of training epochs
    :param train_loader:        torch.utils.data.DataLoader
    :param validation_loader:   torch.utils.data.DataLoader
    :param test_loader:         torch.utils.data.DataLoader
    :param compute_local_loss:  bool, whether to compute the local loss in each layer
    :param update_local_loss:   bool, whether to invoke the optimizer on the local loss in each layer
    :param record_train_acc:    bool
    :param record_val_acc:      bool
    :param record_test_acc:     bool
    :param print_results:       bool, print accuracies if recorded
    :return: np.array of float, np.array of float, np.array of float; train_acc, val_acc, test_acc (in percentage)
    """
    train_acc = -np.ones((n_epochs + 1, 2))
    val_acc = -np.ones((n_epochs + 1, 2))
    test_acc = -np.ones((n_epochs + 1, 2))

    grad_scaler = torch.cuda.amp.GradScaler()

    total_loss_value = 0
    predictor_loss_value = 0

    # todo: make n_locally_connected_layer a class method
    n_locally_connected_layer = np.sum([(isinstance(layer, locally_connected_utils.LocallyConnected2d))
                                  for layer in net.modules()])
    if n_locally_connected_layer > 0:
        record_locally_connected_weight_snr = True
        locally_connected_weight_snr = -np.ones((n_epochs + 1, n_locally_connected_layer))
    else:
        record_locally_connected_weight_snr = False
        locally_connected_weight_snr = None

    # todo: barrier for ddp
    try:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        starting_epoch = checkpoint['epoch'] + 1
        print('Checkpoint loaded at epoch %d to path %s' % (starting_epoch, checkpoint_path))
        del checkpoint
        torch.cuda.empty_cache()
    except:
        # todo: create a checkpoint for rank 0, and load for others (would need to barriers: for save and then for load)
        starting_epoch = 1

    epoch = starting_epoch   # in case we loaded the last epoch
    if epoch > n_epochs:
        warn('epoch > n_epochs, meaning that you restarted a finished simulation')

    hook_handles = locally_connected_utils.register_hebb_hooks(net)

    for epoch in range(starting_epoch, n_epochs + 1):
        if print_results:
            print('Epoch %d on device %s' % (epoch, device))

        if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        total_loss_value = train_one_epoch(net, device, train_loader, optimizer, output_loss,
                                           weight_sharing_training, weight_sharing_frequency, grad_scaler, with_amp,
                                           instant_weight_sharing)

        print('Training losses:\t %s on device %s' % (total_loss_value, device))

        if epoch % stat_report_frequency == 0:
            locally_connected_utils.remove_hebb_hooks(hook_handles)
            record_and_print_running_statistics(record_train_acc, record_val_acc, record_test_acc, print_results,
                                                train_acc, val_acc, test_acc, epoch, net, train_loader,
                                                validation_loader, test_loader, record_locally_connected_weight_snr,
                                                locally_connected_weight_snr, device)
            hook_handles = locally_connected_utils.register_hebb_hooks(net)

        if epoch % checkpoint_frequency == 0:
            # todo: rank constraint + barrier for ddp
            locally_connected_utils.remove_hebb_hooks(hook_handles)

            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print('Checkpoint saved at epoch %d to path %s' % (epoch, checkpoint_path))

            hook_handles = locally_connected_utils.register_hebb_hooks(net)

        if np.any(np.isnan(total_loss_value)) or np.any(np.isnan(predictor_loss_value)):
            warn('NaN during training. total_loss_value: %s; predictor_loss_value: %s' %
                 (total_loss_value, predictor_loss_value))
            return train_acc, val_acc, test_acc, locally_connected_weight_snr

        scheduler.step()

    locally_connected_utils.remove_hebb_hooks(hook_handles)

    # todo: rank constraint + barrier for ddp
    torch.save({
        'epoch': n_epochs + 1,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    print('Checkpoint saved at epoch %d to path %s' % (n_epochs, checkpoint_path))

    return train_acc, val_acc, test_acc, locally_connected_weight_snr


def build_optimizer_and_parameters(opt_name, lr, weight_decay, sgd_momentum, epoch_decrease_lr, opt_lr_decrease):
    """
    Builds an optimizer and parameters for it and for its learning rate scheduler.
    :param opt_name:            str, 'AdamW', 'Adam' or 'SGD'
    :param lr:                  float, initial learning rate
    :param weight_decay:        float
    :param sgd_momentum:        float, momentum used if opt_name == 'SGD'
    :param epoch_decrease_lr:   list of int, epochs when the learning rate decreases
    :param opt_lr_decrease:     float, multiplier by which the learning rate decreases at epoch_decrease_lr
    :return: torch.optim.Optimizer, dict, dict; optimizer, opt_arguments_dict, scheduler_arguments_dict
    """
    opt_arguments_dict = {'lr': lr, 'weight_decay': weight_decay}

    if opt_name == 'AdamW':
        optimizer = optim.AdamW
    elif opt_name == 'Adam':
        optimizer = optim.Adam
    elif 'SGD' in opt_name:
        optimizer = optim.SGD
        opt_arguments_dict.update({'momentum': sgd_momentum, 'nesterov': 'nesterov' in opt_name})
    else:
        raise NotImplementedError('optimizer_local must be either AdamW or Adam or SGD, '
                                  'but %s was passed' % opt_name)
    scheduler_arguments_dict = {'milestones': epoch_decrease_lr, 'gamma': opt_lr_decrease}

    return optimizer, opt_arguments_dict, scheduler_arguments_dict
