import os
import torch
import torchvision
from torchvision import transforms

def dataloader(datasetname : str):
    """ returns dataloaders for a given dataset """

    datasets = { 'celebA': _celebA, 
                 'stanfordcars': _stanfordcars,
                 'mnist': _mnist,
                'cifar10': _cifar10
               }

    return datasets[datasetname.lower()]

def _celebA(
        batchsize : int, testbatchsize : int, datasetfolder : str, 
        augment : bool = False, nworkers : int = 8):
    """ MNIST, 60000 28x28x1 images, 10 classes, 10000 test images """

    transform_totensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CelebA(
        root=datasetfolder, train=True, 
        download=True, transform=transform_totensor)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True, 
        num_workers=nworkers)

    testset = torchvision.datasets.CelebA(
        root=datasetfolder, train=False, 
        download=True, transform=transform_totensor)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=testbatchsize, shuffle=False, 
        num_workers=nworkers)

    return trainset, testset, trainloader, testloader

def _mnist(
        batchsize : int, testbatchsize : int, datasetfolder : str, 
        augment : bool = False, nworkers : int = 8):
    """ MNIST, 60000 28x28x1 images, 10 classes, 10000 test images """

    transform_totensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.MNIST(
        root=datasetfolder, train=True, 
        download=True, transform=transform_totensor)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True, 
        num_workers=nworkers)

    testset = torchvision.datasets.MNIST(
        root=datasetfolder, train=False, 
        download=True, transform=transform_totensor)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=testbatchsize, shuffle=False, 
        num_workers=nworkers)

    return trainset, testset, trainloader, testloader

def _stanfordcars(
        batchsize : int, testbatchsize : int, datasetfolder : str, 
        augment : bool = False, nworkers : int = 8):
    """ FashionMNIST, 60000 28x28x1 images, 10 classes, 10000 test images """

    transform_totensor = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.StanfordCars(
        root=datasetfolder, train=True, 
        download=True, transform=transform_totensor)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batchsize, shuffle=True, 
        num_workers=nworkers)

    testset = torchvision.datasets.StanfordCars(
        root=datasetfolder, train=False, 
        download=True, transform=transform_totensor)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=testbatchsize, shuffle=False, 
        num_workers=nworkers)

    return trainset, testset, trainloader, testloader
