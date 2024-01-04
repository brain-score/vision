import os
import numpy as np

def find_test_accuracy(folder):
    result = []
    for file in os.listdir(folder):
        if 'final' in file and '.out' in file:
            with open(folder + '/' + file, 'r') as f:
                data = f.read().split('\n')

                for i in range(len(data) - 1, -1, -1):
                    if 'CANCELLED' in data[i] or 'IndexError' in data[i]:
                        break
                    if 'Test accuracy (top1, top5):' in data[i]:
                        result.append([float(data[i].split()[4][1:]), float(data[i].split()[5][:-1])])
                        break
    return np.array(result)

def find_running_parameters(folder):
    result = None
    for file in os.listdir(folder):
        if 'final' in file and '.out' in file:
            with open(folder + '/' + file, 'r') as f:
                data = f.read().split('\n')

                for i in range(len(data) - 1, -1, -1):
                    if 'CANCELLED' in data[i] or 'IndexError' in data[i]:
                        break
                    if '--lr' in data[i]:
                        result = [float(data[i].split('--lr ')[1].split()[0]),
                                  float(data[i].split('--weight-decay ')[1].split()[0])]
                        break
    return np.array(result)


def main():
    logs_folder = '/ceph/scratch/romanp/plausible-conv/logs_'
    base_batch_size = 512

    for experiment in ['cifar10', 'cifar100', 'tiny']:
        print(experiment)
        for padding in [0, 4, 8]:
            n_reps = 1
            # conv
            folder = logs_folder + experiment + '/resnet_cifar_%d_3n_%dpad_%dreps' % (base_batch_size // n_reps, padding, n_reps)
            result = find_test_accuracy(folder)
            assert result.shape[0] == 5
            print('conv %d pad over %d trials:\n\tmean: %s\n\tmax-min: %s\n\t' %
                  (padding, result.shape[0], result.mean(axis=0), result.max(axis=0) - result.min(axis=0)))
            # LC
            folder = logs_folder + experiment + '/resnet_cifar_%d_3n_%dpad_%dreps_lc' % (
            base_batch_size // n_reps, padding, n_reps)
            result = find_test_accuracy(folder)
            assert result.shape[0] == 5
            print('LC %d pad over %d trials:\n\tmean: %s\n\tmax-min: %s\n\t' %
                  (padding, result.shape[0], result.mean(axis=0), result.max(axis=0) - result.min(axis=0)))

            if padding > 0:
                for n_reps in [4, 8, 16]:
                    # LC
                    folder = logs_folder + experiment + '/resnet_cifar_%d_3n_%dpad_%dreps_lc' % (
                        base_batch_size // n_reps, padding, n_reps)
                    result = find_test_accuracy(folder)
                    assert result.shape[0] == 5
                    print('LC %d pad %d reps over %d trials:\n\tmean: %s\n\tmax-min: %s\n\t' %
                          (padding, n_reps, result.shape[0], result.mean(axis=0), result.max(axis=0) - result.min(axis=0)))

            n_reps = 1
            # LC + weight sharing
            for w_sh in [1, 10, 100]:
                folder = logs_folder + experiment + '/resnet_cifar_%d_3n_%dpad_%dreps_lc_w_sh_%d_instant' % (
                    base_batch_size // n_reps, padding, n_reps, w_sh)
                result = find_test_accuracy(folder)
                assert result.shape[0] == 5
                print('LC w sh %d, %d pad over %d trials:\n\tmean: %s\n\tmax-min: %s\n\t' %
                      (w_sh, padding, result.shape[0], result.mean(axis=0), result.max(axis=0) - result.min(axis=0)))


def main_table():
    logs_folder = '/ceph/scratch/romanp/plausible-conv/logs_'
    base_batch_size = 512

    row_names = ['conv', 'LC', 'LC - 4 reps', 'LC - 8 reps', 'LC - 16 reps',
                 'LC - ws(1)', 'LC - ws(10)', 'LC - ws(100)']
    row_global_names = ['\multirow{2}{*}{-} ', '\multirow{3}{*}{Data Translation} ', '\multirow{3}{*}{Weight Sharing} ']

    for table in [0, 1]:
        if table == 0:
            result_finder = lambda x: find_test_accuracy(x).mean(axis=0)
        else:
            result_finder = lambda x: find_test_accuracy(x).max(axis=0) - find_test_accuracy(x).min(axis=0)

        for padding in [0, 4, 8]:
            print('PADDING %d' % padding)
        
            final_results = np.zeros((3, 4, 8))  # 3 exp, 4 columns (top1, top5, Dt1, Dt5, 8 setups
        
            for experiment_idx, experiment in enumerate(['cifar10', 'cifar100', 'tiny']):
                    n_reps = 1
                    # conv
                    folder = logs_folder + experiment + '/resnet_cifar_%d_3n_%dpad_%dreps' % (base_batch_size // n_reps, padding, n_reps)
                    result = result_finder(folder)
        
                    final_results[experiment_idx, :2, 0] = result
                    final_results[experiment_idx, 2:, 1:] -= result[:, None]
        
                    # LC
                    folder = logs_folder + experiment + '/resnet_cifar_%d_3n_%dpad_%dreps_lc' % (
                    base_batch_size // n_reps, padding, n_reps)
                    result = result_finder(folder)
        
                    final_results[experiment_idx, :2, 1] = result
                    final_results[experiment_idx, 2:, 1] += result
        
                    if padding > 0:
                        for reps_idx, n_reps in enumerate([4, 8, 16]):
                            # LC
                            folder = logs_folder + experiment + '/resnet_cifar_%d_3n_%dpad_%dreps_lc' % (
                                base_batch_size // n_reps, padding, n_reps)
                            result = result_finder(folder)
                            final_results[experiment_idx, :2, 2 + reps_idx] = result
                            final_results[experiment_idx, 2:, 2 + reps_idx] += result
        
                    n_reps = 1
                    # LC + weight sharing
                    for w_sh_idx, w_sh in enumerate([1, 10, 100]):
                        folder = logs_folder + experiment + '/resnet_cifar_%d_3n_%dpad_%dreps_lc_w_sh_%d_instant' % (
                            base_batch_size // n_reps, padding, n_reps, w_sh)
                        result = result_finder(folder)
                        final_results[experiment_idx, :2, 5 + w_sh_idx] = result
                        final_results[experiment_idx, 2:, 5 + w_sh_idx] += result
        
            for i in range(final_results.shape[-1]):
                if padding > 0 or not (2 <= i <= 4):
                    line = ''
                    for j in range(final_results.shape[0]):
                        if table == 0:
                            k_range = [0, 2, 1, 3]
                        else:
                            k_range = [0, 1]
                        for k in k_range: # top1, then top5
                            if not (j == 0 and k % 2 == 1):  # no top5 for cifar10
                                if final_results[j, k, i] == 0:
                                    line += '& - '
                                else:
                                    line += '& %.1f ' % final_results[j, k, i]
                    if i == 0:
                        print(row_global_names[0] + '& ' + row_names[i] + line + ' \\\\')
                    elif i == 2:
                        print(row_global_names[1] + '& ' + row_names[i] + line + ' \\\\')
                    elif i == 5:
                        print(row_global_names[2] + '& ' + row_names[i] + line + ' \\\\')
                    else:
                        print('& ' + row_names[i] + line + ' \\\\')

    print('PARAMETERS')
    result_finder = lambda x: find_running_parameters(x)
    for padding in [0, 4, 8]:
        print('PADDING %d' % padding)

        final_results = np.zeros((3, 2, 8))  # 3 exp, 4 columns (lr, weight decay), 8 setups

        for experiment_idx, experiment in enumerate(['cifar10', 'cifar100', 'tiny']):
            n_reps = 1
            # conv
            folder = logs_folder + experiment + '/resnet_cifar_%d_3n_%dpad_%dreps' % (
            base_batch_size // n_reps, padding, n_reps)
            result = result_finder(folder)

            final_results[experiment_idx, :2, 0] = result

            # LC
            folder = logs_folder + experiment + '/resnet_cifar_%d_3n_%dpad_%dreps_lc' % (
                base_batch_size // n_reps, padding, n_reps)
            result = result_finder(folder)

            final_results[experiment_idx, :2, 1] = result

            if padding > 0:
                for reps_idx, n_reps in enumerate([4, 8, 16]):
                    # LC
                    folder = logs_folder + experiment + '/resnet_cifar_%d_3n_%dpad_%dreps_lc' % (
                        base_batch_size // n_reps, padding, n_reps)
                    result = result_finder(folder)
                    final_results[experiment_idx, :2, 2 + reps_idx] = result

            n_reps = 1
            # LC + weight sharing
            for w_sh_idx, w_sh in enumerate([1, 10, 100]):
                folder = logs_folder + experiment + '/resnet_cifar_%d_3n_%dpad_%dreps_lc_w_sh_%d_instant' % (
                    base_batch_size // n_reps, padding, n_reps, w_sh)
                result = result_finder(folder)
                final_results[experiment_idx, :2, 5 + w_sh_idx] = result

        for i in range(final_results.shape[-1]):
            if padding > 0 or not (2 <= i <= 4):
                line = ''
                for j in range(final_results.shape[0]):
                    k_range = [0, 1]
                    for k in k_range:  # top1, then top5
                        if final_results[j, k, i] == 0:
                            line += '& - '
                        else:
                            line += '& %g ' % final_results[j, k, i]
                if i == 0:
                    print(row_global_names[0] + '& ' + row_names[i] + line + ' \\\\')
                elif i == 2:
                    print(row_global_names[1] + '& ' + row_names[i] + line + ' \\\\')
                elif i == 5:
                    print(row_global_names[2] + '& ' + row_names[i] + line + ' \\\\')
                else:
                    print('& ' + row_names[i] + line + ' \\\\')

if __name__ == '__main__':
    # main()
    main_table()
