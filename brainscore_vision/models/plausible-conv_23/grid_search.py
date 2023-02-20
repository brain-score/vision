import argparse
import numpy as np
import os
import subprocess
import time


def parse_arguments(args=None):
    """
    Parse the arguments.
    :param args: None or list of str (e.g. ['--device', 'cuda:0']). If None, parses command line arguments. .
    :return: Namespace
    """
    parser = argparse.ArgumentParser(description='Configure the run')

    parser.add_argument("--run-only-final", action='store_true', default=False,
                        help='Run the final simulation, assuming the grid search is done')
    parser.add_argument("--save-final", action='store_true', default=False,
                        help='Save the results of the final run; default: False')
    parser.add_argument("--no-final", action='store_true', default=False,
                        help='Disables final run; default: False')
    parser.add_argument('--n-gpus', type=int, default=1, help='Number of cuda devices to use')
    parser.add_argument('--gpu_name', type=str, default='none',
                        help='GPU type. none or None to use any, otherwise pass the name (e.g. rtx5000); default: none')

    parser.add_argument('--experiment', type=str, default='resnet18_imagenet',
                        help='resnet18_imagenet, resnet_cifar')
    parser.add_argument('--is-locally-connected', action='store_true', default=False,
                        help='Switches conv to locally connected layers; default: False')
    parser.add_argument('--n-first-conv', type=int, default=0,
                        help='How many convolutions in the beginning of the network; default: 0')
    parser.add_argument('--conv-1x1', action='store_true', default=False,
                        help='Switches conv to locally connected layers; default: False')
    parser.add_argument('--freeze-1x1', action='store_true', default=False,
                        help='Switches conv to locally connected layers; default: False')
    parser.add_argument('--lc-conv-start', action='store_true', default=False,
                        help='Conv init for LC layers; default: False')
    parser.add_argument("--dynamic-1x1", action="store_true", default=False,
                        help="Use approximate convolutions; default: False")
    parser.add_argument("--dynamic-NxN", action="store_true", default=False,
                        help="Use approximate convolutions; default: False")
    parser.add_argument("--dynamic-sharing-hebb-lr", type=float, default=0.1,
                        help="Learning rate for dynamic weight sharing in 1x1; default: 0.1")
    parser.add_argument('--dynamic-sharing-b-freq', type=int, default=8, help='Frequency of b generation; default: 8')

    parser.add_argument("--share-weights", action="store_true", default=False,
                        help="Enables weight sharing")
    parser.add_argument("--instant-weight-sharing", action="store_true", default=False,
                        help="Enables weight sharing without actual learning")
    parser.add_argument('--weight-sharing-frequency', type=int, default=1000,
                        help='How often the weights are shared; default: 1000')
    parser.add_argument('--amp', action='store_true', default=False, help='Enables automatic mixed precision')

    # CIFAR
    parser.add_argument('--cifar-image-padding', type=int, default=0,
                        help='Image padding for CIFAR; default: 0')
    parser.add_argument('--cifar-n-reps', type=int, default=1,
                        help='CIFAR number of repetitions within batch; default: 1')
    parser.add_argument('--cifar-resnet-n', type=int, default=3,
                        help='CIFAR resnet depth multiplier (6n+2 layers); default: 3')

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    if args.gpu_name == 'none' or args.gpu_name == 'None':
        args.gpu_name = None

    print("Simulation arguments:", args)

    return args


def make_arguments_grid(fixed_args_dict, grid_args_dict):
    n_runs = 1
    for _, arg_grid in grid_args_dict.items():
        n_runs *= len(arg_grid)

    print('Running %d combinations' % n_runs)

    grid_vals_tuple = tuple()

    for _, arg_grid in grid_args_dict.items():
        grid_vals_tuple += (arg_grid,)

    mesh_grid = np.meshgrid(*grid_vals_tuple, indexing='ij')

    full_args_grid = [fixed_args_dict.copy() for _ in range(n_runs)]

    for run_idx in range(n_runs):
        for arg_idx, arg_name in enumerate(grid_args_dict):
            full_args_grid[run_idx][arg_name] = mesh_grid[arg_idx].flatten()[run_idx]

    print('Full grid:')
    for args in full_args_grid:
        print(args)

    return full_args_grid


def make_bash_command_from_args_dict(args_dict, idx):
    command = 'python3 experiments.py'
    for key, val in args_dict.items():
        command += ' --%s %s' % (key, val)
    command += ' --job-idx %s' % idx
    return command


def count_finished_experiments_in_directory(directory, finish_indicator='Job finished'):
    counter = 0
    for filename in os.listdir(directory):
        if ('.out' in filename) and not ('final' in filename) and not ('makenet' in filename):
            with open(directory + filename, 'r') as file:
                data = file.read()
                if finish_indicator in data:
                    counter += 1
                if 'Error' in data:
                    print('Found error in %s' % (directory + filename))
                    return -1
    return counter


def find_best_accuracy_in_directory(directory, final_acc_indicator='Final validation accuracy: '):
    best_file = ''
    best_acc = 0.0

    for filename in os.listdir(directory):
        if ('.out' in filename) and not ('final' in filename) and not ('makenet' in filename):
            with open(directory + filename, 'r') as file:
                data = file.read()
                data = data.split(final_acc_indicator)
                acc = 0
                n_gpus = 0
                for idx in range(1, len(data), 2):
                    acc += float(data[idx].split('\n')[0])
                    n_gpus += 1
                acc /= n_gpus
                if acc > best_acc:
                    best_acc = acc
                    best_file = filename

    return best_file, best_acc


def run_sbatch(args, job_name, experiment_name, idx, gpu_name=None, logs_folder='logs', n_gpus=1):
    command = make_bash_command_from_args_dict(args, idx)
    sbatch_command = 'sbatch --job-name=\'%s\' --output=\'%s/%s/%s_%%A.out\'' \
                     % (job_name, logs_folder, experiment_name, idx)

    if gpu_name == 'rtx4000':
        sbatch_command += ' --partition=gpu2 --gres=gpu:%s:%d ' \
                          % (gpu_name, n_gpus)
    elif gpu_name == 'a100':
        sbatch_command += ' --partition=a100 --gres=gpu:%s:%d ' \
                          % (gpu_name, n_gpus)
    elif gpu_name is not None:
        sbatch_command += ' --partition=gpu --gres=gpu:%s:%d ' \
                         % (gpu_name, n_gpus)
    else:
        sbatch_command += ' --partition=gpu --gres=gpu:1'

    sbatch_command += ' --export=ALL,COMMAND run_slurm.sh'
    print('Running %s\nCommand: %s' % (sbatch_command, command))

    my_env = os.environ.copy()
    my_env["COMMAND"] = command
    subprocess.run(sbatch_command, shell=True, env=my_env)


def run_grid_search(experiment_name, job_name, fixed_args_dict, grid_args_dict, gpu_name='rtx5000',
                    waiting_time_mins=120, refresh_interval_mins=1, run_only_final=False,
                    logs_folder='logs', n_gpus=1):
    args_grid = make_arguments_grid(fixed_args_dict, grid_args_dict)
    experiments_directory = '%s/%s/' % (logs_folder, experiment_name)
    subprocess.run(['mkdir -p %s' % experiments_directory], shell=True)
    subprocess.run(['mkdir -p %s/checkpoints' % experiments_directory], shell=True)

    if len(args_grid) == 1:
        return args_grid[0]

    if len(args_grid) == count_finished_experiments_in_directory(experiments_directory):
        print('Grid search has already been done')
        filename, accuracy = find_best_accuracy_in_directory(experiments_directory)
        best_args = args_grid[int(filename.split('_')[0])]
        print('Best accuracy: %.2f for \n%s' % (accuracy, best_args))
        return best_args

    if not run_only_final:
        for idx, args in enumerate(args_grid):
            run_sbatch(args, job_name, experiment_name, idx, gpu_name, logs_folder, n_gpus)

    for iter in range(1 + waiting_time_mins // refresh_interval_mins):
        finished_files_counter = count_finished_experiments_in_directory(experiments_directory)
        print('Waiting step %d, finished runs: %d/%d' % (iter, finished_files_counter, len(args_grid)))

        if finished_files_counter == -1:
            raise RuntimeError('Experiments were not finished.')
        if len(args_grid) == finished_files_counter:
            filename, accuracy = find_best_accuracy_in_directory(experiments_directory)
            best_args = args_grid[int(filename.split('_')[0])]
            print('Best accuracy: %.2f for \n%s' % (accuracy, best_args))
            return best_args
        time.sleep(refresh_interval_mins * 60)

    raise RuntimeError('Experiments were not finished.')


def main_imagenet(args):
    is_locally_connected = args.is_locally_connected
    conv_1x1 = args.conv_1x1
    freeze_1x1 = args.freeze_1x1

    fixed_args_dict = {'experiment': args.experiment,
                       'dataset': 'ImageNet', #'ImageNet',
                       'training-mode': 'validation',
                       'validation-size': 10000,
                       'n-epochs': 20,
                       'epoch-decrease-lr': '10 15 ',
                       'locally-connected-deviation-eps': -1, #-1,  # the init weights are totally random
                       'optimizer': 'AdamW',
                       'batch-size': 256 // args.n_gpus,
                       'stat-report-frequency': 10,
                       'checkpoint-frequency': 10,
                       'resnet18-width-per-group': 32,
                       'n-first-conv': args.n_first_conv,
                       'weight-sharing-frequency': args.weight_sharing_frequency,
                       }

    logs_folder = '/ceph/scratch/romanp/plausible-conv/logs_half_r18_correct_val'
    experiment_name = '%s_%d_high_weight_dec' % (fixed_args_dict['experiment'], fixed_args_dict['batch-size'] * args.n_gpus)
    job_name = fixed_args_dict['experiment'][0]

    if is_locally_connected:
        fixed_args_dict['is-locally-connected'] = ''
    if conv_1x1:
        fixed_args_dict['conv-1x1'] = ''
    if freeze_1x1:
        fixed_args_dict['freeze-1x1'] = ''
    if args.amp:
        fixed_args_dict['amp'] = ''
    if args.share_weights:
        fixed_args_dict['share-weights'] = ''
    if args.instant_weight_sharing:
        fixed_args_dict['instant-weight-sharing'] = ''

    grid_args_dict = {'lr': [1e-3, 5e-4],  # 5e-4, 1e-3
                      'weight-decay': [1e-2, 1e-4, 1e-6],# [1e-4],  # [1e-4, 1e-6]
                      }
    if is_locally_connected:
        grid_args_dict = {'lr': [5e-4],
                          'weight-decay': [1e-2],
                          }
    elif not is_locally_connected:
        grid_args_dict = {'lr': [1e-3],
                          'weight-decay': [1e-2],
                          }

    if is_locally_connected:
        experiment_name += '_lc'  # _first_conv
        job_name += '_lc'
        if fixed_args_dict['n-first-conv'] > 0:
            experiment_name += '_%d_first_conv' % fixed_args_dict['n-first-conv']
            job_name += 'f'
        if 'conv-1x1' in fixed_args_dict:
            experiment_name += '_1x1'
            job_name += '1'
        if 'freeze-1x1' in fixed_args_dict:
            experiment_name += '_frozen'
            job_name += 'fr'
        if args.share_weights:
            experiment_name += '_w_sh_%d' % args.weight_sharing_frequency
            if args.instant_weight_sharing:
                experiment_name += '_instant'
            job_name += 'sh%d' % args.weight_sharing_frequency

    fixed_args_dict['experiment-folder'] = '%s/%s' % (logs_folder, experiment_name)

    # make the grids here, run grid search and then run the best one with it
    best_args = run_grid_search(experiment_name, job_name, fixed_args_dict, grid_args_dict,
                                args.gpu_name, waiting_time_mins=10000, run_only_final=args.run_only_final,
                                logs_folder=logs_folder, n_gpus=args.n_gpus)

    for i in range(len(make_arguments_grid(fixed_args_dict, grid_args_dict))):
        subprocess.run(['rm %s/checkpoints/%d.pt' % (fixed_args_dict['experiment-folder'], i)], shell=True)

    best_args['n-epochs'] = 200
    best_args['epoch-decrease-lr'] = '100 150 '
    best_args['training-mode'] = 'test'
    if args.save_final:
        best_args['save-results'] = ''
        best_args['results-filename'] = '%s/%s/final' % (logs_folder, experiment_name)
    run_sbatch(best_args, job_name, experiment_name, idx='final', gpu_name=args.gpu_name,
               logs_folder=logs_folder, n_gpus=args.n_gpus)


def main_cifar(args):

    is_locally_connected = args.is_locally_connected
    conv_1x1 = args.conv_1x1
    freeze_1x1 = args.freeze_1x1

    fixed_args_dict = {'experiment': args.experiment,
                       'dataset': 'CIFAR10',
                       'training-mode': 'validation',
                       'n-epochs': 200,
                       'epoch-decrease-lr': '100 150 ',  # 50 75
                       'locally-connected-deviation-eps': -1, #-1,  # the init weights are totally random
                       'optimizer': 'AdamW',
                       'batch-size': 512 // args.n_gpus // args.cifar_n_reps,
                       'stat-report-frequency': 10,
                       'checkpoint-frequency': 50,
                       'n-first-conv': 0,#args.n_first_conv,
                       'weight-sharing-frequency': args.weight_sharing_frequency,
                       'image-padding': args.cifar_image_padding,
                       'no-cifar10-flip': '',
                       'n-repetitions': args.cifar_n_reps,
                       'resnet-cifar-n': args.cifar_resnet_n,
                       }

    logs_folder = '/ceph/scratch/romanp/plausible-conv/logs_cifar10'
    experiment_name = '%s_%d_%dn_%dpad_%dreps' % (fixed_args_dict['experiment'],
                                                  fixed_args_dict['batch-size'] * args.n_gpus,
                                                  fixed_args_dict['resnet-cifar-n'],
                                                  fixed_args_dict['image-padding'],
                                                  fixed_args_dict['n-repetitions'])
    job_name = fixed_args_dict['experiment'][0] + '%d_%d_%d' % (fixed_args_dict['image-padding'],
                                                                fixed_args_dict['n-repetitions'],
                                                                fixed_args_dict['resnet-cifar-n'],)

    if is_locally_connected:
        fixed_args_dict['is-locally-connected'] = ''
    if conv_1x1:
        fixed_args_dict['conv-1x1'] = ''
    if freeze_1x1:
        fixed_args_dict['freeze-1x1'] = ''
    if args.amp:
        fixed_args_dict['amp'] = ''
    if args.share_weights:
        fixed_args_dict['share-weights'] = ''
    if args.instant_weight_sharing:
        fixed_args_dict['instant-weight-sharing'] = ''

    if is_locally_connected:
        lr_grid = [1e-3, 5e-4, 1e-4, 5e-5]
    else:
        lr_grid = [1e-1, 5e-2, 1e-2, 5e-3]#, 1e-3, 5e-4]

    grid_args_dict = {'lr': lr_grid,  # 5e-4, 1e-3
                      'weight-decay': [1e-2, 1e-4],  # [1e-4, 1e-6]
                      }

    if is_locally_connected:
        experiment_name += '_lc'  # _first_conv
        job_name += '_lc'
        if fixed_args_dict['n-first-conv'] > 0:
            experiment_name += '_%d_first_conv' % fixed_args_dict['n-first-conv']
            job_name += 'f'
        if 'conv-1x1' in fixed_args_dict:
            experiment_name += '_1x1'
            job_name += '1'
        if 'freeze-1x1' in fixed_args_dict:
            experiment_name += '_frozen'
            job_name += 'fr'
        if args.share_weights:
            experiment_name += '_w_sh_%d' % args.weight_sharing_frequency
            if args.instant_weight_sharing:
                experiment_name += '_instant'
            job_name += 'sh%d' % args.weight_sharing_frequency

    fixed_args_dict['experiment-folder'] = '%s/%s' % (logs_folder, experiment_name)

    # make the grids here, run grid search and then run the best one with it
    best_args = run_grid_search(experiment_name, job_name, fixed_args_dict, grid_args_dict,
                                args.gpu_name, waiting_time_mins=10000, run_only_final=args.run_only_final,
                                logs_folder=logs_folder, n_gpus=args.n_gpus)

    for i in range(len(make_arguments_grid(fixed_args_dict, grid_args_dict))):
        subprocess.run(['rm %s/checkpoints/%d.pt' % (fixed_args_dict['experiment-folder'], i)], shell=True)

    best_args['n-epochs'] = 200
    best_args['training-mode'] = 'test'

    if args.save_final:
        best_args['save-results'] = ''
        best_args['results-filename'] = '%s/%s/final' % (logs_folder, experiment_name)

    for i in range(4):
        run_sbatch(best_args, job_name, experiment_name, idx='final_%d' % i, gpu_name=args.gpu_name,
                   logs_folder=logs_folder, n_gpus=args.n_gpus)


def main_cifar100(args):

    is_locally_connected = args.is_locally_connected
    conv_1x1 = args.conv_1x1
    freeze_1x1 = args.freeze_1x1

    fixed_args_dict = {'experiment': args.experiment,
                       'dataset': 'CIFAR100',
                       'training-mode': 'validation',
                       'n-epochs': 200,
                       'epoch-decrease-lr': '100 150 ',  # 50 75
                       'locally-connected-deviation-eps': -1, #-1,  # the init weights are totally random
                       'optimizer': 'AdamW',
                       'batch-size': 512 // args.n_gpus // args.cifar_n_reps,
                       'stat-report-frequency': 10,
                       'checkpoint-frequency': 50,
                       'n-first-conv': 0,#args.n_first_conv,
                       'weight-sharing-frequency': args.weight_sharing_frequency,
                       'image-padding': args.cifar_image_padding,
                       'no-cifar10-flip': '',
                       'n-repetitions': args.cifar_n_reps,
                       'resnet-cifar-n': args.cifar_resnet_n,
                       }

    logs_folder = '/ceph/scratch/romanp/plausible-conv/logs_cifar100'
    experiment_name = '%s_%d_%dn_%dpad_%dreps' % (fixed_args_dict['experiment'],
                                                  fixed_args_dict['batch-size'] * args.n_gpus,
                                                  fixed_args_dict['resnet-cifar-n'],
                                                  fixed_args_dict['image-padding'],
                                                  fixed_args_dict['n-repetitions'])
    job_name = fixed_args_dict['experiment'][0] + '%d_%d_%d' % (fixed_args_dict['image-padding'],
                                                                fixed_args_dict['n-repetitions'],
                                                                fixed_args_dict['resnet-cifar-n'],)

    if is_locally_connected:
        fixed_args_dict['is-locally-connected'] = ''
    if conv_1x1:
        fixed_args_dict['conv-1x1'] = ''
    if freeze_1x1:
        fixed_args_dict['freeze-1x1'] = ''
    if args.amp:
        fixed_args_dict['amp'] = ''
    if args.share_weights:
        fixed_args_dict['share-weights'] = ''
    if args.instant_weight_sharing:
        fixed_args_dict['instant-weight-sharing'] = ''

    if is_locally_connected:
        lr_grid = [1e-3, 5e-4, 1e-4, 5e-5]
    else:
        lr_grid = [1e-1, 5e-2, 1e-2, 5e-3]#, 1e-3, 5e-4]

    grid_args_dict = {'lr': lr_grid,  # 5e-4, 1e-3
                      'weight-decay': [1e-2, 1e-4],  # [1e-4, 1e-6]
                      }

    if is_locally_connected:
        experiment_name += '_lc'  # _first_conv
        job_name += '_lc'
        if fixed_args_dict['n-first-conv'] > 0:
            experiment_name += '_%d_first_conv' % fixed_args_dict['n-first-conv']
            job_name += 'f'
        if 'conv-1x1' in fixed_args_dict:
            experiment_name += '_1x1'
            job_name += '1'
        if 'freeze-1x1' in fixed_args_dict:
            experiment_name += '_frozen'
            job_name += 'fr'
        if args.share_weights:
            experiment_name += '_w_sh_%d' % args.weight_sharing_frequency
            if args.instant_weight_sharing:
                experiment_name += '_instant'
            job_name += 'sh%d' % args.weight_sharing_frequency

    fixed_args_dict['experiment-folder'] = '%s/%s' % (logs_folder, experiment_name)

    # make the grids here, run grid search and then run the best one with it
    best_args = run_grid_search(experiment_name, job_name, fixed_args_dict, grid_args_dict,
                                args.gpu_name, waiting_time_mins=10000, run_only_final=args.run_only_final,
                                logs_folder=logs_folder, n_gpus=args.n_gpus)

    for i in range(len(make_arguments_grid(fixed_args_dict, grid_args_dict))):
        subprocess.run(['rm %s/checkpoints/%d.pt' % (fixed_args_dict['experiment-folder'], i)], shell=True)

    best_args['n-epochs'] = 200
    best_args['training-mode'] = 'test'

    if args.save_final:
        best_args['save-results'] = ''
        best_args['results-filename'] = '%s/%s/final' % (logs_folder, experiment_name)

    for i in range(4):
        run_sbatch(best_args, job_name, experiment_name, idx='final_%d' % i, gpu_name=args.gpu_name,
                   logs_folder=logs_folder, n_gpus=args.n_gpus)


def main_tiny_imagenet(args):
    is_locally_connected = args.is_locally_connected
    conv_1x1 = args.conv_1x1
    freeze_1x1 = args.freeze_1x1

    fixed_args_dict = {'experiment': args.experiment,
                       'dataset': 'TinyImageNet',
                       'training-mode': 'validation',
                       'n-epochs': 50,
                       'epoch-decrease-lr': '25 37 ',  # 50 75
                       'locally-connected-deviation-eps': -1, #-1,  # the init weights are totally random
                       'optimizer': 'AdamW',
                       'batch-size': 512 // args.n_gpus // args.cifar_n_reps,
                       'stat-report-frequency': 10,
                       'checkpoint-frequency': 20,
                       'n-first-conv': 0,#args.n_first_conv,
                       'weight-sharing-frequency': args.weight_sharing_frequency,
                       'image-padding': args.cifar_image_padding,
                       'no-cifar10-flip': '',
                       'n-repetitions': args.cifar_n_reps,
                       'resnet-cifar-n': args.cifar_resnet_n,
                       }

    logs_folder = '/ceph/scratch/romanp/plausible-conv/logs_tiny'
    experiment_name = '%s_%d_%dn_%dpad_%dreps' % (fixed_args_dict['experiment'],
                                                  fixed_args_dict['batch-size'] * args.n_gpus,
                                                  fixed_args_dict['resnet-cifar-n'],
                                                  fixed_args_dict['image-padding'],
                                                  fixed_args_dict['n-repetitions'])
    job_name = 't_%d_%d_%d' % (fixed_args_dict['image-padding'],
                               fixed_args_dict['n-repetitions'],
                               fixed_args_dict['resnet-cifar-n'],)

    if is_locally_connected:
        fixed_args_dict['is-locally-connected'] = ''
    if conv_1x1:
        fixed_args_dict['conv-1x1'] = ''
    if freeze_1x1:
        fixed_args_dict['freeze-1x1'] = ''
    if args.amp:
        fixed_args_dict['amp'] = ''
    if args.share_weights:
        fixed_args_dict['share-weights'] = ''
    if args.instant_weight_sharing:
        fixed_args_dict['instant-weight-sharing'] = ''

    if is_locally_connected:
        lr_grid = [1e-3, 5e-4] # seem to be useless, 1e-4, 5e-5]
    else:
        lr_grid = [5e-3, 1e-3, 5e-4]#[1e-2, 5e-3, 1e-3, 5e-4]#, 1e-3, 5e-4]

    grid_args_dict = {'lr': lr_grid,  # 5e-4, 1e-3
                      'weight-decay': [1e-2, 1e-4], #[1e-2, 1e-4],  # [1e-4, 1e-6]
                      }

    if is_locally_connected:
        experiment_name += '_lc'  # _first_conv
        job_name += '_lc'
        if fixed_args_dict['n-first-conv'] > 0:
            experiment_name += '_%d_first_conv' % fixed_args_dict['n-first-conv']
            job_name += 'f'
        if 'conv-1x1' in fixed_args_dict:
            experiment_name += '_1x1'
            job_name += '1'
        if 'freeze-1x1' in fixed_args_dict:
            experiment_name += '_frozen'
            job_name += 'fr'
        if args.share_weights:
            experiment_name += '_w_sh_%d' % args.weight_sharing_frequency
            if args.instant_weight_sharing:
                experiment_name += '_instant'
            job_name += 'sh%d' % args.weight_sharing_frequency

    fixed_args_dict['experiment-folder'] = '%s/%s' % (logs_folder, experiment_name)

    # make the grids here, run grid search and then run the best one with it
    best_args = run_grid_search(experiment_name, job_name, fixed_args_dict, grid_args_dict,
                                args.gpu_name, waiting_time_mins=10000, run_only_final=args.run_only_final,
                                logs_folder=logs_folder, n_gpus=args.n_gpus)

    for i in range(len(make_arguments_grid(fixed_args_dict, grid_args_dict))):
        subprocess.run(['rm %s/checkpoints/%d.pt' % (fixed_args_dict['experiment-folder'], i)], shell=True)

    best_args['epoch-decrease-lr'] = '100 150 '
    best_args['n-epochs'] = 200
    best_args['training-mode'] = 'test'

    if args.save_final:
        best_args['save-results'] = ''
        best_args['results-filename'] = '%s/%s/final' % (logs_folder, experiment_name)

    if fixed_args_dict['image-padding'] != 4:
        for i in range(5):
            run_sbatch(best_args, job_name, experiment_name, idx='final_%d' % i, gpu_name=args.gpu_name,
                       logs_folder=logs_folder, n_gpus=args.n_gpus)
    elif fixed_args_dict['image-padding'] == 4:
        for i in range(4):
            run_sbatch(best_args, job_name, experiment_name, idx='final_%d' % i, gpu_name=args.gpu_name,
                       logs_folder=logs_folder, n_gpus=args.n_gpus)
    else:
        print('\n\n\nFinal runs for padding > 4 are disabled for now\n\n\n')
        # run_sbatch(best_args, job_name, experiment_name, idx='final', gpu_name=args.gpu_name,
        #            logs_folder=logs_folder, n_gpus=args.n_gpus)


def main_mbv3(args):
    is_locally_connected = args.is_locally_connected
    conv_1x1 = args.conv_1x1
    freeze_1x1 = args.freeze_1x1

    if args.lc_conv_start:
        lc_eps = 0
    else:
        lc_eps = -1

    fixed_args_dict = {'experiment': args.experiment,
                       'dataset': 'ImageNet', #'ImageNet',
                       'training-mode': 'validation',
                       'validation-size': 10000,
                       'n-epochs': 40,
                       'epoch-decrease-lr': '20 30 ',
                       'locally-connected-deviation-eps': lc_eps, #-1,  # the init weights are totally random
                       'optimizer': 'AdamW',  # 'SGD_nesterov', #
                       'sgd-momentum': 0.9,
                       'batch-size': 1024 // args.n_gpus,
                       'stat-report-frequency': 10,
                       'checkpoint-frequency': 10,
                       'n-first-conv': args.n_first_conv,
                       'weight-sharing-frequency': args.weight_sharing_frequency,
                       'imagenet-path': '/tmp/roman/imagenet/'
                       }

    logs_folder = '/ceph/scratch/romanp/plausible-conv/logs_mbv3_cosyne'
    experiment_name = '%s_%d' % (fixed_args_dict['experiment'], fixed_args_dict['batch-size'] * args.n_gpus)
    job_name = fixed_args_dict['experiment'][0]

    if is_locally_connected:
        fixed_args_dict['is-locally-connected'] = ''
    if conv_1x1:
        fixed_args_dict['conv-1x1'] = ''
    if freeze_1x1:
        fixed_args_dict['freeze-1x1'] = ''
    if args.amp:
        fixed_args_dict['amp'] = ''
    if args.share_weights:
        fixed_args_dict['share-weights'] = ''
    if args.instant_weight_sharing:
        fixed_args_dict['instant-weight-sharing'] = ''
    if args.dynamic_NxN and is_locally_connected:
        fixed_args_dict['dynamic-NxN'] = ''
        fixed_args_dict['dynamic-sharing-hebb-lr'] = args.dynamic_sharing_hebb_lr
        fixed_args_dict['dynamic-sharing-b-freq'] = args.dynamic_sharing_b_freq
    if args.dynamic_1x1 and is_locally_connected:
        fixed_args_dict['dynamic-1x1'] = ''
        fixed_args_dict['dynamic-sharing-hebb-lr'] = args.dynamic_sharing_hebb_lr
        fixed_args_dict['dynamic-sharing-b-freq'] = args.dynamic_sharing_b_freq

    # batch size 512
    grid_args_dict = {'lr': [5e-3, 1e-3, 5e-4], # sgd: [5e-1, 1e-1, 5e-2],   # adam: 5e-3, 1e-3, 5e-4
                      'weight-decay': [1e-2], # sgd: [1e-6],  # LC: 1e-2
                      }

    if (args.dynamic_1x1 or args.dynamic_NxN) and is_locally_connected:
        # grid_args_dict = {'lr': [1e-3],
        #                   'weight-decay': [1e-2],  # LC: 1e-2
        #                   'dynamic-sharing-hebb-lr': [1e-3], #[1e-1, 1e-2, 1e-3, 1e-4, 1e-5],  # [1e3, 5e2, 1e2, 5e1, 1e1, 5, 0.1],
        #                   }
        grid_args_dict = {'lr': [1e-3],
                          'weight-decay': [1e-2],  # LC: 1e-2
                          # 'dynamic-sharing-hebb-lr': [1e3, 5e2, 1e2, 5e1, 1e1, 5, 0.1], # no wd + adamw + normal b
                          # #[1e-3, 1e-4, 1e-5],
                          # # [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],  # [1e3, 5e2, 1e2, 5e1, 1e1, 5, 0.1],
                          }


    if is_locally_connected:
        experiment_name += '_lc'  # _first_conv
        job_name += '_lc'
        if fixed_args_dict['n-first-conv'] > 0:
            experiment_name += '_%d_first_conv' % fixed_args_dict['n-first-conv']
            job_name += 'f'
        if 'conv-1x1' in fixed_args_dict:
            experiment_name += '_1x1'
            job_name += '1'
        if 'freeze-1x1' in fixed_args_dict:
            experiment_name += '_frozen'
            job_name += 'fr'
        if args.share_weights:
            experiment_name += '_w_sh_%d' % args.weight_sharing_frequency
            if args.instant_weight_sharing:
                experiment_name += '_instant'
            job_name += 'sh%d' % args.weight_sharing_frequency
        if args.lc_conv_start:
            experiment_name += '_conv_start'
        if args.dynamic_NxN:
            experiment_name += '_dyn_NxN'
        if args.dynamic_1x1:
            experiment_name += '_dyn_1x1'

    fixed_args_dict['experiment-folder'] = '%s/%s' % (logs_folder, experiment_name)

    # make the grids here, run grid search and then run the best one with it
    best_args = run_grid_search(experiment_name, job_name, fixed_args_dict, grid_args_dict,
                                args.gpu_name, waiting_time_mins=10000, run_only_final=args.run_only_final,
                                logs_folder=logs_folder, n_gpus=args.n_gpus)

    for i in range(len(make_arguments_grid(fixed_args_dict, grid_args_dict))):
        subprocess.run(['rm %s/checkpoints/%d.pt' % (fixed_args_dict['experiment-folder'], i)], shell=True)

    # if best_args['lr'] == 5e-4 and (best_args['weight-sharing-frequency'] == 10 or
    #                                 best_args['weight-sharing-frequency'] == 100):
    #     subprocess.run(['rm %s/checkpoints/final.pt' % fixed_args_dict['experiment-folder']], shell=True)

    best_args['n-epochs'] = 200  # 150
    best_args['epoch-decrease-lr'] = '100 150 '  # '75 118 '
    best_args['training-mode'] = 'test'
    if args.save_final:
        best_args['save-results'] = ''
        best_args['results-filename'] = '%s/%s/final' % (logs_folder, experiment_name)
    if not args.no_final:
        run_sbatch(best_args, job_name, experiment_name, idx='final_200', gpu_name=args.gpu_name,
                   logs_folder=logs_folder, n_gpus=args.n_gpus)


def main_resnet18(args):
    is_locally_connected = args.is_locally_connected
    conv_1x1 = args.conv_1x1
    freeze_1x1 = args.freeze_1x1

    fixed_args_dict = {'experiment': args.experiment,
                       'dataset': 'ImageNet', #'ImageNet',
                       'training-mode': 'validation',
                       'validation-size': 10000,
                       'n-epochs': 20,
                       'epoch-decrease-lr': '10 15 ',
                       'locally-connected-deviation-eps': -1, #-1,  # the init weights are totally random
                       'optimizer': 'AdamW',
                       'batch-size': 256 // args.n_gpus,
                       'stat-report-frequency': 10,
                       'checkpoint-frequency': 10,
                       'resnet18-width-per-group': 32,
                       'n-first-conv': args.n_first_conv,
                       'weight-sharing-frequency': args.weight_sharing_frequency,
                       }

    logs_folder = '/ceph/scratch/romanp/plausible-conv/logs_half_r18_no_first_conv'#_correction'
    experiment_name = '%s_%d_high_weight_dec' % (fixed_args_dict['experiment'], fixed_args_dict['batch-size'] * args.n_gpus)
    job_name = fixed_args_dict['experiment'][0]

    if is_locally_connected:
        fixed_args_dict['is-locally-connected'] = ''
    if conv_1x1:
        fixed_args_dict['conv-1x1'] = ''
    if freeze_1x1:
        fixed_args_dict['freeze-1x1'] = ''
    if args.amp:
        fixed_args_dict['amp'] = ''
    if args.share_weights:
        fixed_args_dict['share-weights'] = ''
    if args.instant_weight_sharing:
        fixed_args_dict['instant-weight-sharing'] = ''

    grid_args_dict = {'lr': [1e-3, 5e-4],  # 5e-4, 1e-3
                      'weight-decay': [1e-2, 1e-4, 1e-6],# [1e-4],  # [1e-4, 1e-6]
                      }
    if is_locally_connected:
        grid_args_dict = {'lr': [1e-3, 5e-4], #[1e-4, 5e-5], #[1e-3, 5e-4],
                          'weight-decay': [1e-2],
                          }
    elif not is_locally_connected:
        grid_args_dict = {'lr': [1e-3],
                          'weight-decay': [1e-2],
                          }

    if is_locally_connected:
        experiment_name += '_lc'  # _first_conv
        job_name += '_lc'
        if fixed_args_dict['n-first-conv'] > 0:
            experiment_name += '_%d_first_conv' % fixed_args_dict['n-first-conv']
            job_name += 'f'
        if 'conv-1x1' in fixed_args_dict:
            experiment_name += '_1x1'
            job_name += '1'
        if 'freeze-1x1' in fixed_args_dict:
            experiment_name += '_frozen'
            job_name += 'fr'
        if args.share_weights:
            experiment_name += '_w_sh_%d' % args.weight_sharing_frequency
            if args.instant_weight_sharing:
                experiment_name += '_instant'
            job_name += 'sh%d' % args.weight_sharing_frequency

    fixed_args_dict['experiment-folder'] = '%s/%s' % (logs_folder, experiment_name)

    # make the grids here, run grid search and then run the best one with it
    best_args = run_grid_search(experiment_name, job_name, fixed_args_dict, grid_args_dict,
                                args.gpu_name, waiting_time_mins=10000, run_only_final=args.run_only_final,
                                logs_folder=logs_folder, n_gpus=args.n_gpus)

    for i in range(len(make_arguments_grid(fixed_args_dict, grid_args_dict))):
        subprocess.run(['rm %s/checkpoints/%d.pt' % (fixed_args_dict['experiment-folder'], i)], shell=True)

    best_args['n-epochs'] = 200
    best_args['epoch-decrease-lr'] = '100 150 '
    best_args['training-mode'] = 'test'
    if args.save_final:
        best_args['save-results'] = ''
        best_args['results-filename'] = '%s/%s/final' % (logs_folder, experiment_name)
    # if not args.no_final:
    run_sbatch(best_args, job_name, experiment_name, idx='final', gpu_name=args.gpu_name,
               logs_folder=logs_folder, n_gpus=args.n_gpus)


def main_full_resnet18(args):
    is_locally_connected = args.is_locally_connected
    conv_1x1 = args.conv_1x1
    freeze_1x1 = args.freeze_1x1

    fixed_args_dict = {'experiment': args.experiment,
                       'dataset': 'ImageNet', #'ImageNet',
                       'training-mode': 'validation',
                       'validation-size': 10000,
                       'n-epochs': 20,
                       'epoch-decrease-lr': '10 15 ',
                       'locally-connected-deviation-eps': -1, #-1,  # the init weights are totally random
                       'optimizer': 'AdamW',
                       'batch-size': 128 // args.n_gpus,
                       'stat-report-frequency': 10,
                       'checkpoint-frequency': 10,
                       'resnet18-width-per-group': 64,
                       'n-first-conv': args.n_first_conv,
                       'weight-sharing-frequency': args.weight_sharing_frequency,
                       }

    logs_folder = '/ceph/scratch/romanp/plausible-conv/logs_r18_no_first_conv'
    experiment_name = '%s_%d_high_weight_dec' % (fixed_args_dict['experiment'], fixed_args_dict['batch-size'] * args.n_gpus)
    job_name = fixed_args_dict['experiment'][0]

    if is_locally_connected:
        fixed_args_dict['is-locally-connected'] = ''
    if conv_1x1:
        fixed_args_dict['conv-1x1'] = ''
    if freeze_1x1:
        fixed_args_dict['freeze-1x1'] = ''
    if args.amp:
        fixed_args_dict['amp'] = ''
    if args.share_weights:
        fixed_args_dict['share-weights'] = ''
    if args.instant_weight_sharing:
        fixed_args_dict['instant-weight-sharing'] = ''

    grid_args_dict = {'lr': [1e-3, 5e-4, 1e-4, 5e-5],
                      'weight-decay': [1e-2],
                      }

    if is_locally_connected:
        experiment_name += '_lc'  # _first_conv
        job_name += '_lc'
        if fixed_args_dict['n-first-conv'] > 0:
            experiment_name += '_%d_first_conv' % fixed_args_dict['n-first-conv']
            job_name += 'f'
        if 'conv-1x1' in fixed_args_dict:
            experiment_name += '_1x1'
            job_name += '1'
        if 'freeze-1x1' in fixed_args_dict:
            experiment_name += '_frozen'
            job_name += 'fr'
        if args.share_weights:
            experiment_name += '_w_sh_%d' % args.weight_sharing_frequency
            if args.instant_weight_sharing:
                experiment_name += '_instant'
            job_name += 'sh%d' % args.weight_sharing_frequency

    fixed_args_dict['experiment-folder'] = '%s/%s' % (logs_folder, experiment_name)

    # make the grids here, run grid search and then run the best one with it
    best_args = run_grid_search(experiment_name, job_name, fixed_args_dict, grid_args_dict,
                                args.gpu_name, waiting_time_mins=10000, run_only_final=args.run_only_final,
                                logs_folder=logs_folder, n_gpus=args.n_gpus)

    for i in range(len(make_arguments_grid(fixed_args_dict, grid_args_dict))):
        subprocess.run(['rm %s/checkpoints/%d.pt' % (fixed_args_dict['experiment-folder'], i)], shell=True)

    best_args['n-epochs'] = 200
    best_args['epoch-decrease-lr'] = '100 150 '
    best_args['training-mode'] = 'test'
    if args.save_final:
        best_args['save-results'] = ''
        best_args['results-filename'] = '%s/%s/final' % (logs_folder, experiment_name)
    # if not args.no_final:
    run_sbatch(best_args, job_name, experiment_name, idx='final', gpu_name=args.gpu_name,
               logs_folder=logs_folder, n_gpus=args.n_gpus)


def main_cornet_s(args):
    is_locally_connected = args.is_locally_connected
    conv_1x1 = args.conv_1x1
    freeze_1x1 = args.freeze_1x1

    if args.lc_conv_start:
        lc_eps = 0
    else:
        lc_eps = -1

    fixed_args_dict = {'experiment': args.experiment,
                       'dataset': 'ImageNet', #'ImageNet',
                       'training-mode': 'validation',
                       'validation-size': 10000,
                       'n-epochs': 200,
                       'epoch-decrease-lr': '100 150 ',
                       'locally-connected-deviation-eps': lc_eps, #-1,  # the init weights are totally random
                       'optimizer': 'AdamW',
                       'batch-size': 384 // args.n_gpus,
                       'stat-report-frequency': 10,
                       'checkpoint-frequency': 10,
                       'n-first-conv': args.n_first_conv,
                       'weight-sharing-frequency': args.weight_sharing_frequency,
                       'imagenet-path': '/tmp/roman/imagenet/',
                       'cornet-scale': 1,
                       }

    logs_folder = '/ceph/scratch/romanp/plausible-conv/logs_cornet_s_%d_cosyne' % fixed_args_dict['cornet-scale']
    experiment_name = '%s_%d' % (fixed_args_dict['experiment'], fixed_args_dict['batch-size'] * args.n_gpus)
    job_name = fixed_args_dict['experiment'][0]

    if is_locally_connected:
        fixed_args_dict['is-locally-connected'] = ''
    if conv_1x1:
        fixed_args_dict['conv-1x1'] = ''
    if freeze_1x1:
        fixed_args_dict['freeze-1x1'] = ''
    if args.amp:
        fixed_args_dict['amp'] = ''
    if args.share_weights:
        fixed_args_dict['share-weights'] = ''
    if args.instant_weight_sharing:
        fixed_args_dict['instant-weight-sharing'] = ''
    if args.dynamic_1x1 and is_locally_connected:
        fixed_args_dict['dynamic-1x1'] = ''
        fixed_args_dict['dynamic-sharing-hebb-lr'] = args.dynamic_sharing_hebb_lr
        fixed_args_dict['dynamic-sharing-b-freq'] = args.dynamic_sharing_b_freq
    if args.dynamic_NxN and is_locally_connected:
        fixed_args_dict['dynamic-NxN'] = ''
        fixed_args_dict['dynamic-sharing-hebb-lr'] = args.dynamic_sharing_hebb_lr
        fixed_args_dict['dynamic-sharing-b-freq'] = args.dynamic_sharing_b_freq

    # batch size 512
    grid_args_dict = {'lr': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
                      'weight-decay': [1e-2, 1e-4],  # LC: 1e-2
                      }
    if is_locally_connected:
        # # low wd is bad, high lrs lead to divergence
        # grid_args_dict = {'lr': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
        #                   'weight-decay': [1e-2, 1e-4],  # LC: 1e-2
        #                   }
        grid_args_dict = {'lr':  [5e-4, 1e-4, 5e-5], # [5e-4, 1e-4, 5e-5, 1e-5],
                          'weight-decay': [1e-1, 1e-2],  # LC: 1e-2
                          }

        if args.n_first_conv > 0:
            grid_args_dict = {'lr': [5e-3, 1e-3, 5e-4, 1e-4],
                              'weight-decay': [1e-1, 1e-2],  # LC: 1e-2
                              }
    if is_locally_connected:
        experiment_name += '_lc'  # _first_conv
        job_name += '_lc'
        if fixed_args_dict['n-first-conv'] > 0:
            experiment_name += '_%d_first_conv' % fixed_args_dict['n-first-conv']
            job_name += 'f'
        if 'conv-1x1' in fixed_args_dict:
            experiment_name += '_1x1'
            job_name += '1'
        if 'freeze-1x1' in fixed_args_dict:
            experiment_name += '_frozen'
            job_name += 'fr'
        if args.share_weights:
            experiment_name += '_w_sh_%d' % args.weight_sharing_frequency
            if args.instant_weight_sharing:
                experiment_name += '_instant'
            job_name += 'sh%d' % args.weight_sharing_frequency
        if args.lc_conv_start:
            experiment_name += '_conv_start'
        if args.dynamic_1x1:
            experiment_name += '_dyn_1x1'
        if args.dynamic_NxN:
            experiment_name += '_dyn_NxN'

    fixed_args_dict['experiment-folder'] = '%s/%s' % (logs_folder, experiment_name)

    # make the grids here, run grid search and then run the best one with it
    best_args = run_grid_search(experiment_name, job_name, fixed_args_dict, grid_args_dict,
                                args.gpu_name, waiting_time_mins=10000, run_only_final=args.run_only_final,
                                logs_folder=logs_folder, n_gpus=args.n_gpus)

    for i in range(len(make_arguments_grid(fixed_args_dict, grid_args_dict))):
        subprocess.run(['rm %s/checkpoints/%d.pt' % (fixed_args_dict['experiment-folder'], i)], shell=True)

    # if best_args['lr'] == 5e-4 and (best_args['weight-sharing-frequency'] == 10 or
    #                                 best_args['weight-sharing-frequency'] == 100):
    #     subprocess.run(['rm %s/checkpoints/final.pt' % fixed_args_dict['experiment-folder']], shell=True)

    best_args['n-epochs'] = 200  # 400
    best_args['epoch-decrease-lr'] = '100 150 '  # '200 300 '  #
    best_args['training-mode'] = 'test'
    if args.save_final:
        best_args['save-results'] = ''
        best_args['results-filename'] = '%s/%s/final' % (logs_folder, experiment_name)
    if not args.no_final:
        run_sbatch(best_args, job_name, experiment_name, idx='final_200', gpu_name=args.gpu_name,
                   logs_folder=logs_folder, n_gpus=args.n_gpus)


def main_resnet18_dyn(args):
    is_locally_connected = args.is_locally_connected
    conv_1x1 = args.conv_1x1
    freeze_1x1 = args.freeze_1x1

    if args.lc_conv_start:
        lc_eps = 0
    else:
        lc_eps = -1

    fixed_args_dict = {'experiment': args.experiment,
                       'dataset': 'ImageNet', #'ImageNet',
                       'training-mode': 'validation',
                       'validation-size': 10000,
                       'n-epochs': 40,
                       'epoch-decrease-lr': '20 30 ',
                       'locally-connected-deviation-eps': lc_eps, #-1,  # the init weights are totally random
                       'optimizer': 'AdamW',  # 'SGD_nesterov', #
                       'sgd-momentum': 0.9,
                       'batch-size': 512 // args.n_gpus,
                       'stat-report-frequency': 10,
                       'checkpoint-frequency': 10,
                       'resnet18-width-per-group': 32,
                       'n-first-conv': args.n_first_conv,
                       'weight-sharing-frequency': args.weight_sharing_frequency,
                       'imagenet-path': '/tmp/roman/imagenet/'
                       }

    logs_folder = '/ceph/scratch/romanp/plausible-conv/logs_half_r18_cosyne'
    experiment_name = '%s_%d' % (fixed_args_dict['experiment'], fixed_args_dict['batch-size'] * args.n_gpus)
    job_name = fixed_args_dict['experiment'][0]

    if is_locally_connected:
        fixed_args_dict['is-locally-connected'] = ''
    if conv_1x1:
        fixed_args_dict['conv-1x1'] = ''
    if freeze_1x1:
        fixed_args_dict['freeze-1x1'] = ''
    if args.amp:
        fixed_args_dict['amp'] = ''
    if args.share_weights:
        fixed_args_dict['share-weights'] = ''
    if args.instant_weight_sharing:
        fixed_args_dict['instant-weight-sharing'] = ''
    if args.dynamic_NxN and is_locally_connected:
        fixed_args_dict['dynamic-NxN'] = ''
    if args.dynamic_1x1 and is_locally_connected:
        fixed_args_dict['dynamic-1x1'] = ''
        # fixed_args_dict['dynamic-sharing-hebb-lr'] = args.dynamic_sharing_hebb_lr
        fixed_args_dict['dynamic-sharing-b-freq'] = args.dynamic_sharing_b_freq

    # batch size 512
    grid_args_dict = {'lr': [5e-3, 1e-3, 5e-4], # sgd: [5e-1, 1e-1, 5e-2],   # adam: 5e-3, 1e-3, 5e-4
                      'weight-decay': [1e-2], # sgd: [1e-6],  # LC: 1e-2
                      }

    if args.dynamic_1x1 and is_locally_connected:
        grid_args_dict = {'lr': [1e-3, 5e-4],
                          'weight-decay': [1e-2],  # LC: 1e-2
                          'dynamic-sharing-hebb-lr': [1e-3, 1e-4, 1e-5],  # [1e3, 5e2, 1e2, 5e1, 1e1, 5, 0.1],
                          }
        # SGD:
        # grid_args_dict = {'lr': [1e-1],
        #                   'weight-decay': [1e-6],  # LC: 1e-2
        #                   'dynamic-sharing-hebb-lr': [1e3, 1e2, 1e1, 1],
        #                   }

    if is_locally_connected:
        experiment_name += '_lc'  # _first_conv
        job_name += '_lc'
        if fixed_args_dict['n-first-conv'] > 0:
            experiment_name += '_%d_first_conv' % fixed_args_dict['n-first-conv']
            job_name += 'f'
        if 'conv-1x1' in fixed_args_dict:
            experiment_name += '_1x1'
            job_name += '1'
        if 'freeze-1x1' in fixed_args_dict:
            experiment_name += '_frozen'
            job_name += 'fr'
        if args.share_weights:
            experiment_name += '_w_sh_%d' % args.weight_sharing_frequency
            if args.instant_weight_sharing:
                experiment_name += '_instant'
            job_name += 'sh%d' % args.weight_sharing_frequency
        if args.lc_conv_start:
            experiment_name += '_conv_start'
        if args.dynamic_NxN:
            experiment_name += '_dyn_NxN'
        if args.dynamic_1x1:
            experiment_name += '_dyn_1x1_noiseless_sgd_b'

    fixed_args_dict['experiment-folder'] = '%s/%s' % (logs_folder, experiment_name)

    # make the grids here, run grid search and then run the best one with it
    best_args = run_grid_search(experiment_name, job_name, fixed_args_dict, grid_args_dict,
                                args.gpu_name, waiting_time_mins=10000, run_only_final=args.run_only_final,
                                logs_folder=logs_folder, n_gpus=args.n_gpus)

    for i in range(len(make_arguments_grid(fixed_args_dict, grid_args_dict))):
        subprocess.run(['rm %s/checkpoints/%d.pt' % (fixed_args_dict['experiment-folder'], i)], shell=True)

    # if best_args['lr'] == 5e-4 and (best_args['weight-sharing-frequency'] == 10 or
    #                                 best_args['weight-sharing-frequency'] == 100):
    #     subprocess.run(['rm %s/checkpoints/final.pt' % fixed_args_dict['experiment-folder']], shell=True)

    best_args['n-epochs'] = 200  # 150
    best_args['epoch-decrease-lr'] = '100 150 '  # '75 118 '
    best_args['training-mode'] = 'test'
    if args.save_final:
        best_args['save-results'] = ''
        best_args['results-filename'] = '%s/%s/final' % (logs_folder, experiment_name)
    if not args.no_final:
        run_sbatch(best_args, job_name, experiment_name, idx='final_200', gpu_name=args.gpu_name,
                   logs_folder=logs_folder, n_gpus=args.n_gpus)


if __name__ == '__main__':
    args = parse_arguments()
    print('Running with %s' % args)
    if args.experiment == 'resnet18_imagenet' and args.dynamic_NxN:
        # main_imagenet(args)
        main_resnet18_dyn(args)
    elif args.experiment == 'resnet18_imagenet':
        # main_imagenet(args)
        main_resnet18(args)
    elif args.experiment == 'full_resnet18_imagenet':
        main_full_resnet18(args)
    elif args.experiment == 'resnet_cifar10':
        args.experiment = 'resnet_cifar'  # todo: fix in experiments
        main_cifar(args)
        # main_cifar_long(args)
        # main_cifar_scaled(args)
        # main_cifar100(args)
    elif args.experiment == 'resnet_cifar100':
        args.experiment = 'resnet_cifar'  # todo: fix in experiments
        # main_cifar(args)
        # main_cifar_long(args)
        # main_cifar_scaled(args)
        main_cifar100(args)
    elif args.experiment == 'TinyImageNet':
        args.experiment = 'resnet_cifar'  # todo: fix in experiments
        main_tiny_imagenet(args)
    elif args.experiment == 'mobilenetv3_imagenet':
        main_mbv3(args)
    elif args.experiment == 'cornet_s_imagenet':
        main_cornet_s(args)
