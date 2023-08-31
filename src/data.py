import torch
import numpy as np
import copy

import torchvision  # type: ignore

mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def load_dataset(dataset_name):
    transforms_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]
    )

    transforms_test = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean, std)]
    )

    if dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(
            root="./", train=True, transform=transforms_train, download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./", train=False, transform=transforms_test, download=True
        )
    else:
        train_dataset = torchvision.datasets.CIFAR100(
            root="./", train=True, transform=transforms_train, download=True
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root="./", train=False, transform=transforms_test, download=True
        )

    return train_dataset, test_dataset


def get_loaders(train_dataset, test_dataset, batch_size):
    # Decide on the sizes of the splits. Here, let's assume 80% training, 20% validation
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    # Use random_split to split the dataset into training and validation datasets
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, valid_loader, test_loader


def task_construction(task_labels, dataset_name, order=None):
    train_dataset, test_dataset = load_dataset(dataset_name)

    train_dataset.targets = torch.tensor(train_dataset.targets)
    test_dataset.targets = torch.tensor(test_dataset.targets)

    if order is not None:
        train_targets = -1 * torch.tensor(
            np.ones(len(train_dataset.targets)), dtype=torch.long
        )
        test_targets = -1 * torch.tensor(
            np.ones(len(test_dataset.targets)), dtype=torch.long
        )
        for i, label in enumerate(order):
            train_targets[train_dataset.targets == label] = i
            test_targets[test_dataset.targets == label] = i

        train_dataset.targets = train_targets.clone()
        test_dataset.targets = test_targets.clone()

    train_dataset = split_dataset_by_labels(train_dataset, task_labels)
    test_dataset = split_dataset_by_labels(test_dataset, task_labels)
    return train_dataset, test_dataset


def split_dataset_by_labels(dataset, task_labels):
    datasets = []
    for labels in task_labels:
        idx = np.in1d(dataset.targets, labels)
        splited_dataset = copy.deepcopy(dataset)
        splited_dataset.targets = torch.tensor(splited_dataset.targets)[idx]
        splited_dataset.data = splited_dataset.data[idx]
        datasets.append(splited_dataset)
    return datasets


def create_labels(num_classes, num_tasks, num_classes_per_task):
    tasks_order = np.arange(num_classes)
    labels = tasks_order.reshape((num_tasks, num_classes_per_task))
    return labels


def create_svhn_dataset(root_dir, transform, split="train"):
    return torchvision.datasets.SVHN(
        root=root_dir, split=split, download=True, transform=transform
    )


def create_svhn_dataloader(dataset, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2
    )
