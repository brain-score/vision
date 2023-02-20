import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler, Sampler, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from PIL import Image
from warnings import warn
import os


class SubsetDeterministicSampler(Sampler):
    """
    Samples elements non-randomly from a given list of indices.
    """
    def __init__(self, indices):
        """
        :param indices: list, tuple, np.array or torch.Tensor of ints, a sequence of indices
        """
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.arange(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class SubsetDistributedSampler(DistributedSampler):
    def __init__(self, dataset, subset_indices, num_replicas=None, rank=None, shuffle=True, seed=0):
        super().__init__(dataset, num_replicas, rank, shuffle, seed)
        self.subset_indices = subset_indices
        self.num_samples = int(np.ceil(len(self.subset_indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.subset_indices), generator=g).tolist()
        else:
            indices = list(range(len(self.subset_indices)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(self.subset_indices[indices])


class MultiplePointCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, n_repetitions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_repetitions = n_repetitions

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            images = tuple(self.transform(img) for i in range(self.n_repetitions))
        else:
            images = tuple(img.copy() for i in range(self.n_repetitions))
        if self.target_transform is not None:
            target = self.target_transform(target)

        return images + (target,)


class MultiplePointCIFAR10Subset(MultiplePointCIFAR10):
    def __init__(self, indices, n_repetitions, *args, **kwargs):
        super().__init__(n_repetitions, *args, **kwargs)
        self.data = self.data[indices]
        self.targets = np.array(self.targets)[indices]


class CIFAR10Subset(torchvision.datasets.CIFAR10):
    def __init__(self, indices, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.data[indices]
        self.targets = np.array(self.targets)[indices]


class MultiplePointCIFAR100(torchvision.datasets.cifar.CIFAR100):
    def __init__(self, n_repetitions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_repetitions = n_repetitions

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            images = tuple(self.transform(img) for i in range(self.n_repetitions))
        else:
            images = tuple(img.copy() for i in range(self.n_repetitions))
        if self.target_transform is not None:
            target = self.target_transform(target)

        return images + (target,)


class MultiplePointCIFAR100Subset(MultiplePointCIFAR100):
    def __init__(self, indices, n_repetitions, *args, **kwargs):
        super().__init__(n_repetitions, *args, **kwargs)
        self.data = self.data[indices]
        self.targets = np.array(self.targets)[indices]


class CIFAR100Subset(torchvision.datasets.cifar.CIFAR100):
    def __init__(self, indices, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = self.data[indices]
        self.targets = np.array(self.targets)[indices]


class MultiplePointImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, n_repetitions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_repetitions = n_repetitions

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            images = tuple(self.transform(img) for i in range(self.n_repetitions))
        else:
            images = tuple(img.copy() for i in range(self.n_repetitions))
        if self.target_transform is not None:
            target = self.target_transform(target)

        return images + (target,)


def build_cifar10_transforms(mean, std, padding=4, do_flip=True, input_scale=1):
    """
    Crops, flips and normalizes train data, normalizes test data.
    :return:    torchvision.transforms, torchvision.transforms; transform_train, transform_test
    """
    # mean = (0.4914, 0.4822, 0.4465)  # https://github.com/kuangliu/pytorch-cifar/issues/19
    adjusted_mean = (255 * np.array(mean)).astype(int) / 255
    fill_values = tuple((255 * adjusted_mean).astype(int))  # makes sure the filled values are zero after normalization
    # std = (0.247, 0.243, 0.261)

    # normalization before cropping to have the same distribution

    transform_train = list()
    transform_test = list()

    image_size = int(32 * input_scale)

    if input_scale !=1:
        transform_train.append(transforms.Resize(image_size))
        transform_test.append(transforms.Resize(image_size))

    transform_train.append(transforms.RandomCrop(image_size, padding=padding, fill=fill_values))
    if do_flip:
        transform_train.append(transforms.RandomHorizontalFlip())

    transform_train.append(transforms.ToTensor())
    transform_train.append(transforms.Normalize(tuple(adjusted_mean), std))
    transform_test.append(transforms.ToTensor())
    transform_test.append(transforms.Normalize(tuple(adjusted_mean), std))

    transform_train = transforms.Compose(transform_train)
    transform_test = transforms.Compose(transform_test)

    return transform_train, transform_test


def find_training_set_mean_std(torch_dataset, validation_ratio, train_validation_split_seed, input_scale=1):
    if (torch_dataset == torchvision.datasets.mnist.MNIST and
            validation_ratio == 0.1 and train_validation_split_seed == 0):
        return (0.13067308,), (0.308122,)
    if (torch_dataset == torchvision.datasets.cifar.CIFAR10 and
            validation_ratio == 0.1 and train_validation_split_seed == 0 and input_scale == 1):
        return (0.49147674, 0.48228467, 0.44654816), (0.24709952, 0.2434634, 0.2615838)
    if (torch_dataset == torchvision.datasets.cifar.CIFAR100 and
                validation_ratio == 0.1 and train_validation_split_seed == 0 and input_scale == 1):
        return (0.5065967, 0.48631412, 0.44057995), (0.26733804, 0.2565225, 0.27617523)
    if torch_dataset == torchvision.datasets.cifar.CIFAR10 and validation_ratio == 0.0 and input_scale == 1:
        return (0.49139965, 0.4821585, 0.44653103), (0.24703225, 0.24348518, 0.26158783)

    if input_scale == 1:
        transform_test = transforms.ToTensor()
    else:
        assert torch_dataset == torchvision.datasets.cifar.CIFAR10 or \
            torch_dataset == torchvision.datasets.cifar.CIFAR100, 'Input scale != 1 only for for CIFAR'
        transform_test = transforms.Compose([transforms.Resize(int(32 * input_scale)),
                                             transforms.ToTensor()])

    full_train_set = torch_dataset(root='./data', train=True, download=True, transform=transform_test)
    if validation_ratio > 0.0:
        indices = np.arange(len(full_train_set), dtype=int)
        np.random.RandomState(train_validation_split_seed).shuffle(indices)

        split_size = int(validation_ratio * len(full_train_set))

        sampler = SubsetDeterministicSampler(indices[split_size:])
    else:
        sampler = SequentialSampler(full_train_set)

    train_loader = torch.utils.data.DataLoader(full_train_set, batch_size=128,
                                               num_workers=2, sampler=sampler, pin_memory=True)

    mean = None
    n_iter = 0
    for data in train_loader:
        if mean is not None:
            mean += data[0].mean(dim=(0, 2, 3)) * data[0].shape[0]
        else:
            mean = data[0].mean(dim=(0, 2, 3)) * data[0].shape[0]
        n_iter += data[0].shape[0]
    mean = mean / n_iter

    std = torch.zeros_like(mean)
    for data in train_loader:
        std += ((data[0] - mean[None, :, None, None]) ** 2).mean(dim=(0, 2, 3)) * data[0].shape[0]
    std = (std / n_iter) ** 0.5
    print('Dataset centering details:')
    print(torch_dataset, validation_ratio, train_validation_split_seed, tuple(mean.numpy()), tuple(std.numpy()))
    return tuple(mean.numpy()), tuple(std.numpy())


def build_loaders(torch_dataset, build_transforms, padding=4, n_repetitions=1,
                  batch_size=128, validation_ratio=0.1, train_validation_split_seed=0,
                  rank=None, world_size=1, input_scale=1):
    """
    Builds data loaders for the given dataset.
    :param torch_dataset:               str, 'CIFAR10', 'MNIST', 'fMNIST' (fashion-MNIST) or 'kMNIST' (Kuzushiji-MNIST)
    :param build_transforms:            function that returns two lists of torchvision.transforms (train and test)
    :param batch_size:                  int, batch size
    :param validation_ratio:            float, validation size / full train set size.
        If zero, validation set is the full train set but without random transformations (e.g. crops)
    :param train_validation_split_seed: int, numpy random seed for train/validation split
    :return: torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader;
        train_loader, valid_loader, test_loader
    """
    train_mean, train_std = find_training_set_mean_std(torch_dataset, validation_ratio, train_validation_split_seed,
                                                       input_scale)
    transform_train, transform_test = build_transforms(train_mean, train_std, padding, input_scale)

    if torch_dataset == torchvision.datasets.CIFAR10:
        full_train_class = MultiplePointCIFAR10
        train_class = MultiplePointCIFAR10Subset
        val_class = CIFAR10Subset
    elif torch_dataset == torchvision.datasets.cifar.CIFAR100:
        full_train_class = MultiplePointCIFAR100
        train_class = MultiplePointCIFAR100Subset
        val_class = CIFAR100Subset
    else:
        raise NotImplementedError('Dataset must be CIFAR10 or CIFAR100, but %s was given' % torch_dataset)

    if validation_ratio > 0.0:
        full_train_set = full_train_class(n_repetitions, root='data', train=True, transform=transform_train,
                                          download=True)

        indices = np.arange(len(full_train_set), dtype=int)
        np.random.RandomState(train_validation_split_seed).shuffle(indices)
        split_size = int(validation_ratio * len(full_train_set))

        train_set = train_class(indices[split_size:], n_repetitions, root='data', train=True,
                                transform=transform_train, download=True)
        validation_set = val_class(indices[:split_size], root='data', train=True,
                                   transform=transform_test, download=True)
    else:
        print('\nNo validation, validation data will be the training set with test transforms\n')
        train_set = full_train_class(n_repetitions, root='data', train=True,
                                     transform=transform_train, download=True)
        validation_set = torch_dataset(root='data', train=True, transform=transform_test, download=True)

    test_set = torch_dataset(root='data', train=False, transform=transform_test, download=True)

    if world_size > 1:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        valid_sampler = DistributedSampler(validation_set, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   sampler=train_sampler, num_workers=8, pin_memory=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                                        sampler=valid_sampler, num_workers=8,
                                                        pin_memory=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                  sampler=test_sampler, num_workers=8, pin_memory=True)
        return train_loader, validation_loader, test_loader

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=8, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                                    shuffle=False, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, validation_loader, test_loader


def build_imagenet_transforms(dataset, padding=0):
    if dataset == 'ImageNet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transforms_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transforms_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    elif dataset == 'TinyImageNet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        transforms_train = transforms.Compose([
            transforms.CenterCrop(48 + 2 * padding),
            transforms.RandomCrop(48),
            transforms.ToTensor(),
            normalize,
        ])

        transforms_test = transforms.Compose([
            transforms.CenterCrop(48),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise NotImplementedError('dataset must be ImageNet or TinyImageNet, but %s was given' % dataset)

    return transforms_train, transforms_test


def build_imagenet_loaders(batch_size=128, rank=None, world_size=1,
                           imagenet_path='/nfs/gatsbystor/michaela/imagenet/images',  # '/ceph/scratch/romanp/imagenet',
                           train_validation_split_seed=0, validation_size=10000, dataset='ImageNet', padding=0,
                           n_repetitions=1):
    train_dir = os.path.join(imagenet_path, 'train')
    val_dir = os.path.join(imagenet_path, 'val')

    transforms_train, transforms_test = build_imagenet_transforms(dataset, padding)

    if n_repetitions == 1:
        train_set = torchvision.datasets.ImageFolder(train_dir, transforms_train)
    else:
        train_set = MultiplePointImageFolder(n_repetitions, train_dir, transforms_train)
    validation_set = torchvision.datasets.ImageFolder(train_dir, transforms_test)
    test_set = torchvision.datasets.ImageFolder(val_dir, transforms_test)

    if validation_size > 0:
        train_set = torch.utils.data.random_split(
            train_set, [len(train_set) - validation_size, validation_size],
            generator=torch.Generator().manual_seed(train_validation_split_seed))[0]
        validation_set = torch.utils.data.random_split(
            validation_set, [len(validation_set) - validation_size, validation_size],
            generator=torch.Generator().manual_seed(train_validation_split_seed))[1]

    if world_size > 1:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        valid_sampler = DistributedSampler(validation_set, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        valid_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                                               shuffle=(train_sampler is None), num_workers=16,
                                               pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,
                                                    sampler=valid_sampler, shuffle=(valid_sampler is None),
                                                    num_workers=16, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              sampler=test_sampler, shuffle=(test_sampler is None),
                                              num_workers=16, pin_memory=True)

    return train_loader, validation_loader, test_loader


def build_loaders_by_dataset(dataset, batch_size=128, padding=0, n_repetitions=1,
                             validation_ratio=0.1, train_validation_split_seed=0,
                             do_cifar10_flip=True, rank=None, world_size=1, input_scale=1, validation_size=10000,
                             imagenet_path=None):
    """
    Builds loaders for a specific dataset
    :param dataset:                     str, 'CIFAR10', 'MNIST', 'fMNIST' (fashion-MNIST) or 'kMNIST' (Kuzushiji-MNIST)
    :param batch_size:                  int, batch size
    :param validation_ratio:            float, validation size / full train set size.
        If zero, validation set is the full train set but without random transformations (e.g. crops)
    :param train_validation_split_seed: int, numpy random seed for train/validation split
    :return: torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader;
        train_loader, valid_loader, test_loader
    """
    if dataset == 'CIFAR10':
        return build_loaders(torchvision.datasets.CIFAR10,
                             lambda x, y, z, s: build_cifar10_transforms(x, y, z, do_cifar10_flip, s),
                             padding, n_repetitions, batch_size, validation_ratio, train_validation_split_seed,
                             rank=rank, world_size=world_size, input_scale=input_scale)
    if dataset == 'CIFAR100':
        return build_loaders(torchvision.datasets.cifar.CIFAR100,
                             lambda x, y, z, s: build_cifar10_transforms(x, y, z, do_cifar10_flip, s),
                             padding, n_repetitions, batch_size, validation_ratio, train_validation_split_seed,
                             rank=rank, world_size=world_size, input_scale=input_scale)
    elif dataset == 'ImageNet':
        # todo: add imagenet folder argument
        if padding != 0:
            raise NotImplementedError('ImageNet only works for padding=0, but %d was given' % padding)
        if input_scale != 1:
            raise NotImplementedError('ImageNet only works for input_scale=1, but %f was given' % input_scale)
        if imagenet_path is None:
            imagenet_path = '/nfs/gatsbystor/michaela/imagenet/images'
        return build_imagenet_loaders(batch_size, rank, world_size,
                                      train_validation_split_seed=train_validation_split_seed,
                                      validation_size=validation_size, n_repetitions=n_repetitions,
                                      imagenet_path=imagenet_path)
    elif dataset == 'TinyImageNet':
        # todo: add imagenet folder argument
        if input_scale != 1:
            raise NotImplementedError('ImageNet only works for input_scale=1, but %f was given' % input_scale)
        if imagenet_path is None:
            imagenet_path = '/tmp/romanp/tiny-imagenet-200'
        return build_imagenet_loaders(batch_size, rank, world_size,
                                      train_validation_split_seed=train_validation_split_seed,
                                      validation_size=validation_size, dataset='TinyImageNet',
                                      imagenet_path=imagenet_path, padding=padding,
                                      n_repetitions=n_repetitions)
    else:
        raise NotImplementedError('dataset must be either CIFAR10, or MNIST, or kMNIST, or fMNIST, '
                                  'but %s was given' % dataset)
