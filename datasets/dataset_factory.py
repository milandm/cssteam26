import torchvision
import torchvision.transforms as transforms
import torchvision.datasets
from datasets.cifar10c import CIFAR10C


def build_dataset(dataset_name, dataset_path, train=True, args=None):
    # cifar10
    if dataset_name == "cifar10":
        # setup transforms
        if train:
            transform = transforms.Compose([
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        dataset = torchvision.datasets.CIFAR10(dataset_path, transform=transform, train=train, download=False)
        return dataset

    # cifar10-c
    if dataset_name == "cifar10c":
        cifar10c_corruptions_config_key = 'corruptions'
        cifar10c_severities_config_key = 'severities'
        if not (args and cifar10c_corruptions_config_key in args and cifar10c_severities_config_key in args):
            exit(f"cifar10c requires {cifar10c_corruptions_config_key}"
                 f" and {cifar10c_severities_config_key} config keys to be specified")
        corruptions = args[cifar10c_corruptions_config_key]
        severities = args[cifar10c_severities_config_key]

        # setup transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset = CIFAR10C(dataset_path,
                           corruptions=corruptions,
                           severities=severities,
                           train=train,
                           transform=transform)
        return dataset

    if dataset_name == "imagenet":
        # setup transforms
        transform = transforms.Compose([
            # transforms.RandomResizedCrop(config.get("resolution")),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
        return dataset

    exit(f'{dataset_name} dataset is not supported')
