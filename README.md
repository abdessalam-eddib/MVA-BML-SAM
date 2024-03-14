# MVA-BML-SAM

## Bayesian Machine Learning Project

This repository contains the code implementation for the Bayesian Machine Learning project based on the paper "SAM as an Optimal Relaxation of Bayes" [link to the paper]. 

### Overview

We provide an adaptation of the official implementation of the SAM (Sharpness-Aware Minimization) algorithm, allowing for training the ResNet18 neural network on various datasets using different optimizers: SGD, SAM, and bSAM.

### Usage

#### 2D Classification with bSAM Optimizer

To demonstrate the functionality of the bSAM optimizer, we start by classifying data in a 2D space. The script `bSAM_classifier.py` is provided for this purpose. We have tested it on the `make_moons` and `make_circles` datasets. Results with the hyperparameters used are presented in the corresponding folder. Users can modify the file to experiment with different hyperparameters and datasets. To execute the script, use the following command:

```bash
%run bSAM_classifier.py
```

#### General Training with ResNet18

For more general training, ResNet18 can be trained on CIFAR10 or any other dataset listed in the `data.py` file. Use the following commands:

- Using SGD optimizer:
```bash
%run train.py --alpha 0.5 --beta1 0.9 --beta2 0.999 --priorprec 40 --rho 0.01 --batchsplit 8 --optim sgd --dataset cifar10 --dafactor 4 --epochs 20
```
Execution time: 1048s, Accuracy: 92.39%

- Using SAM optimizer:
```bash
%run train.py --alpha 0.5 --beta1 0.9 --beta2 0.999 --priorprec 40 --rho 0.01 --batchsplit 8 --optim sam --dataset cifar10 --dafactor 4 --epochs 20
```
Execution time: 2976s, Accuracy: 92.67%

- Using bSAM optimizer:
```bash
%run train.py --alpha 0.5 --beta1 0.9 --beta2 0.999 --priorprec 40 --rho 0.01 --batchsplit 8 --optim bsam --dataset cifar10 --dafactor 4 --epochs 20
```
Execution time: 3592s, Accuracy: 92.96%

### Performance Visualization

We provide visualizations of the training loss and accuracy over the 20 training epochs.

![Training Loss and Accuracy Plot](path/to/image)

### Pretrained Models

Due to GitHub limitations, our pretrained models are stored on Google Drive. You can access them from the following link: [link to Google Drive].

### Dependencies

- Python 3.x
- PyTorch
- NumPy
- Matplotlib
- Other dependencies as listed in `requirements.txt`

### References

- SAM: Sharpness-Aware Minimization: [link to the paper]

For any questions or issues, feel free to contact [maintainer's email].
 



