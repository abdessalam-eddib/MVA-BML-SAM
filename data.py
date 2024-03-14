import os
import torch
import torchvision
from torchvision import transforms

def dataloader(datasetname : str):
    """ returns dataloaders for a given dataset """

    datasets = {'cifar10': _cifar10,
                'food101': _food101,
                'stanfordcars': _stanfordcars,
                'fakedata': _fakedata,
               }

    return datasets[datasetname.lower()]




def _cifar10(
        batchsize : int, testbatchsize : int, datasetfolder : str, 
        augment : bool = True, nworkers : int = 8):
    """ CIFAR-10, 50000 32x32x3 images, 10 classes, 10000 test images """

    train_mean = (0.4914, 0.4822, 0.4465)
    train_std = (0.2023, 0.1994, 0.2010)

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=datasetfolder, train=True, 
        download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True, 
        num_workers=nworkers, drop_last=True)

    testset = torchvision.datasets.CIFAR10(
        root=datasetfolder, train=False, 
        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=testbatchsize, shuffle=False, 
        num_workers=nworkers, drop_last=False)

    return trainset, testset, trainloader, testloader

    
def _food101(
        batchsize: int, testbatchsize: int, datasetfolder: str,
        augment: bool = True, nworkers: int = 8):
    """ Food101, 101 food categories, variable resolution images """

    train_mean = (0.5, 0.5, 0.5)
    train_std = (0.5, 0.5, 0.5)

    # Define transformations for data augmentation and normalization
    if augment:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
    ])

    trainset = torchvision.datasets.Food101(
        root=datasetfolder, 
        download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize,
        shuffle=True, num_workers=nworkers, drop_last=True)

    testset = torchvision.datasets.Food101(
        root=datasetfolder, split="test", 
        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=testbatchsize,
        shuffle=False, num_workers=nworkers, drop_last=False)

    return trainset, testset, trainloader, testloader


def _stanfordcars(
        batchsize: int, testbatchsize: int, datasetfolder: str,
        augment: bool = True, nworkers: int = 8):
    """ Stanford Cars, 196 makes, 8144 models, 16,185 images """

    train_mean = (0.5, 0.5, 0.5)
    train_std = (0.5, 0.5, 0.5)

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
    ])

    trainset = torchvision.datasets.StanfordCars(
        root=datasetfolder, 
        download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize,
        shuffle=True, num_workers=nworkers, drop_last=True)

    testset = torchvision.datasets.StanfordCars(
        root=datasetfolder, split="test", 
        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=testbatchsize,
        shuffle=False, num_workers=nworkers, drop_last=False)

    return trainset, testset, trainloader, testloader


def _fakedata(
        batchsize: int, testbatchsize: int, datasetfolder: str,
        augment: bool = True, nworkers: int = 8):
    """ FakeData, randomly generated data for debugging """

    trainset = torchvision.datasets.FakeData(
        size=50000, image_size=(3, 224, 224), num_classes=10,
        transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize,
        shuffle=True, num_workers=nworkers, drop_last=True)

    testset = torchvision.datasets.FakeData(
        size=10000, image_size=(3, 224, 224), num_classes=10,
        transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=testbatchsize,
        shuffle=False, num_workers=nworkers, drop_last=False)

    return trainset, testset, trainloader, testloader


def _caltech101(
        batchsize: int, testbatchsize: int, datasetfolder: str,
        augment: bool = True, nworkers: int = 8):
    """ Caltech101, 101 object categories, 9144 images """

    train_mean = (0.5, 0.5, 0.5)
    train_std = (0.5, 0.5, 0.5)

    if augment:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std),
    ])

    trainset = torchvision.datasets.ImageFolder(
        os.path.join(datasetfolder, 'caltech-101', 'train'), transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize,
        shuffle=True, num_workers=nworkers, drop_last=True)

    testset = torchvision.datasets.ImageFolder(
        os.path.join(datasetfolder, 'caltech-101', 'test'), transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=testbatchsize,
        shuffle=False, num_workers=nworkers, drop_last=False)

    return trainset, testset, trainloader, testloader
