""" 1. Train

Train a new network on a data set with train.py

    Basic usage: python train.py data_directory
    Prints out training loss, validation loss, and validation accuracy as the network trains
    Options: 
    * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory 
    * Choose architecture: python train.py data_dir --arch "vgg13" 
    * Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 
    * Use GPU for training: python train.py data_dir --gpu """

import argparse
import json
import torch
from collections import OrderedDict
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg11(weights=False)
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 2048)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.2)),
        ('fc2', nn.Linear(2048, 512)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(0.2)),
        ('fc3', nn.Linear(512, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier   
    model.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


parser = argparse.ArgumentParser(description='Train and save an image classification model.')
parser.add_argument('data_dir', help='The directory of images you want the model to be trained on', type=str)

args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

if torch.cuda.is_available():
    print('CUDA used:', torch.cuda.memory_allocated())
    device = torch.device('cuda')

if data_dir:
    # transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform=val_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=64)
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    
    model = load_checkpoint('checkpoint.pth')
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    
    # training
    epochs = 2
    for epoch in range(epochs):
        print('EPOCH', epoch)
    
        # notify the model we are in training mode
        model.train()
        train_loss = 0
        val_loss   = 0
    
        print('Training', end='')
        for n, (images, labels) in enumerate(trainloader):
            # move images and label tensors to the default device
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()        
            train_loss += loss.item()
            if not n % 20:
                print('.', end='', flush=True)
        print(n, 'loops')
    
        # notify the model we are in eval mode
        # remember no_grad() for the work
        model.eval()
        print('Validating', end='')
        with torch.no_grad():
            accuracy = 0
            for n, (images, labels) in enumerate(valloader):
                images, labels = images.to(device), labels.to(device)
                logps = model.forward(images)
                batch_loss = criterion(logps, labels)
                val_loss   = batch_loss.item()
            
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                if not n % 10:
                    print('.', end='', flush=True)
            print(n, 'loops')
        
        train_loss = train_loss/len(trainloader)
        val_loss   = val_loss/len(valloader)

        val_accuracy = accuracy/len(valloader)

        # print statistics
        print(f'Training Loss: {train_loss:.3f}')
        print(f'Validation Loss: {val_loss:.3f}')
        print(f'Validation Accuracy: {val_accuracy:.3f}')
