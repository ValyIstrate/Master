from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import v2
import torch


def get_augmentation(augmentation_type):
    if augmentation_type == "basic":
        return transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    elif augmentation_type == "advanced":
        return transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor()])
    else:
        return transforms.ToTensor()


def get_dataloader(config):
    augmentation = get_augmentation(config['dataset']['augmentation'])
    basic_transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
    ])

    train_transforms = v2.Compose([basic_transforms, augmentation])

    dataset_name = config['dataset']['name']
    dataset_path = config['dataset']['path']

    if dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root=dataset_path, train=True, transform=basic_transforms, download=True)
        test_dataset = datasets.MNIST(root=dataset_path, train=False, transform=basic_transforms, download=True)
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root=dataset_path, train=True, transform=basic_transforms, download=True)
        test_dataset = datasets.CIFAR10(root=dataset_path, train=False, transform=basic_transforms, download=True)
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(root=dataset_path, train=True, transform=basic_transforms, download=True)
        test_dataset = datasets.CIFAR100(root=dataset_path, train=False, transform=basic_transforms, download=True)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")

    train_dataset = SimpleCachedDataset(train_dataset)
    test_dataset = SimpleCachedDataset(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True,
                              num_workers=config['dataset']['num_workers'], pin_memory=config['dataset']['pin_memory'])

    test_loader = DataLoader(test_dataset, batch_size=config['dataset']['batch_size'], shuffle=False,
                             num_workers=config['dataset']['num_workers'], pin_memory=config['dataset']['pin_memory'])

    return train_loader, test_loader


class SimpleCachedDataset(Dataset):
    def __init__(self, dataset):
        self.data = tuple([x for x in dataset])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]
