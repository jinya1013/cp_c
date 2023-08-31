import torch
import numpy as np
import copy

import torchvision  # type: ignore

from torch.utils.data import Dataset, DataLoader

from typing import List, Tuple, Final, Optional, Union

mean: Final[Tuple[float, float, float]] = (0.485, 0.456, 0.406)
std: Final[Tuple[float, float, float]] = (0.229, 0.224, 0.225)


class CifarDataset(Dataset):
    base_folder: str
    url: str
    filename: str
    tgz_md5: str
    train_list: List[Tuple[str, str]]
    test_list: List[Tuple[str, str]]
    data: np.ndarray
    targets: Union[List[int], torch.Tensor]


# def create_labels(
#     num_classes: int, num_tasks: int, num_classes_per_task: int
# ) -> np.ndarray:
#     tasks_order = np.arange(num_classes)
#     labels = tasks_order.reshape((num_tasks, num_classes_per_task))
#     return labels


def create_labels(num_classes: int, classes_per_task: List[int]) -> List[np.ndarray]:
    """
    Create labels for each task

    Parameters
    ----------
    num_classes : int
        the number of classes in the dataset
    classes_per_task : List[int]
        the number of classes in each task

    Returns
    -------
    List[np.ndarray]
        a list of arrays containing the labels for each task
    """
    assert (
        sum(classes_per_task) == num_classes
    ), "Total classes in all tasks should match num_classes"

    tasks_order = np.arange(num_classes)
    labels = []

    start_idx = 0
    for num_classes_in_task in classes_per_task:
        end_idx = start_idx + num_classes_in_task
        task_labels = tasks_order[start_idx:end_idx]
        labels.append(task_labels)
        start_idx = end_idx

    return labels


def load_dataset(
    dataset_name: str,
) -> Tuple[CifarDataset, CifarDataset]:
    """
    Load the dataset

    Parameters
    ----------
    dataset_name : str
        the name of the dataset, cifar10 or cifar100

    Returns
    -------
    Tuple[Dataset, Dataset]
        a tuple containing the train and test datasets

    Raises
    ------
    ValueError
        if the dataset_name is not supported (not cifar10 or cifar100)
    """
    transforms_train = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
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
    elif dataset_name == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root="./", train=True, transform=transforms_train, download=True
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root="./", train=False, transform=transforms_test, download=True
        )
    else:
        raise ValueError("Dataset not supported")

    return train_dataset, test_dataset


def get_loaders(
    train_dataset: CifarDataset,
    test_dataset: CifarDataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader,]:
    """
    Get the train, validation and test loaders from the datasets

    Parameters
    ----------
    train_dataset : Dataset
    test_dataset : Dataset
    batch_size : int

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader,]
        a tuple containing the train, validation and test loaders
    """
    # Decide on the sizes of the splits. Here, let's assume 80% training, 20% validation
    train_size = int(0.8 * len(train_dataset))
    valid_size = len(train_dataset) - train_size

    # Use random_split to split the dataset into training and validation datasets
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    valid_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return train_loader, valid_loader, test_loader


def split_dataset_by_labels(
    dataset: CifarDataset, task_labels: List[np.ndarray]
) -> List[CifarDataset]:
    """
    Split the dataset into multiple datasets based on the labels

    Parameters
    ----------
    dataset : CifarDataset
        Cifar10 or Cifar100 dataset
    task_labels : List[np.ndarray]
        task_labels[i] contains the labels for task i

    Returns
    -------
    List[CifarDataset]
        a list of datasets, each containing the data for a task
    """
    datasets = []
    for labels in task_labels:
        idx = list(np.in1d(dataset.targets, labels))  # ラベルが一致するインデックスを取得
        split_dataset = copy.deepcopy(dataset)
        split_dataset.targets = torch.tensor(split_dataset.targets)[idx]
        split_dataset.data = split_dataset.data[idx]
        datasets.append(split_dataset)
    return datasets


def task_construction(
    task_labels: List[np.ndarray], dataset_name: str, order: Optional[str] = None
) -> Tuple[List[CifarDataset], List[CifarDataset]]:
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


import unittest


class TestCIFARFunctions(unittest.TestCase):
    def test_load_dataset_cifar10(self):
        train_dataset, test_dataset = load_dataset("cifar10")
        self.assertEqual(len(train_dataset), 50000)
        self.assertEqual(len(test_dataset), 10000)

    def test_load_dataset_cifar100(self):
        train_dataset, test_dataset = load_dataset("cifar100")
        self.assertEqual(len(train_dataset), 50000)
        self.assertEqual(len(test_dataset), 10000)

    def test_create_labels(self):
        labels = create_labels(10, [2, 3, 5])
        expected = [np.array([0, 1]), np.array([2, 3, 4]), np.array([5, 6, 7, 8, 9])]
        for l, e in zip(labels, expected):
            self.assertTrue(np.array_equal(l, e))

    def test_get_loaders(self):
        train_dataset, test_dataset = load_dataset("cifar10")
        train_loader, valid_loader, test_loader = get_loaders(
            train_dataset, test_dataset, 32
        )
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(valid_loader)
        self.assertIsNotNone(test_loader)

    def test_split_dataset_by_labels(self):
        train_dataset, _ = load_dataset("cifar10")
        task_labels = [np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9])]
        split_datasets = split_dataset_by_labels(train_dataset, task_labels)
        self.assertEqual(len(split_datasets[0]), 25000)  # Roughly half for each label
        self.assertEqual(len(split_datasets[1]), 25000)

    def test_task_construction(self):
        task_labels = [np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9])]
        train_datasets, test_datasets = task_construction(task_labels, "cifar10")
        self.assertEqual(len(train_datasets[0]), 25000)  # Roughly half for each label
        self.assertEqual(len(train_datasets[1]), 25000)
        self.assertEqual(len(test_datasets[0]), 5000)  # Roughly half for each label
        self.assertEqual(len(test_datasets[1]), 5000)


if __name__ == "__main__":
    unittest.main()
