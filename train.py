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
import os
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


def save_model(save_dir):
    checkpoint = {
        'input_size': [3, 224, 224],
        'output_size': 102,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'class_to_idx': train_data.class_to_idx
    }
    save_path = f'{save_dir}/saved_model.pth'
    torch.save(checkpoint, save_path)
    print(f'Model saved to {save_path}')


def train_transformer(train_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    return train_data


def test_transformer(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data


def val_transformer(val_dir):
    val_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    # Load the Data
    val_data = datasets.ImageFolder(valid_dir, transform=val_transforms)
    return val_data


def data_loader(data, batch_size=64, shuffle=False):
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return loader


def create_model(architecture):
    print(f'\t----------\nUsing {architecture} pretrained network...\n\t----------')  
    if architecture == 'densenet161':
        model = models.densenet161(pretrained=True)
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)   
    
    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False
        
    return model



def check_save_dir(save_dir):
    if os.path.isdir(save_dir):
        return True
    else:
        print("\tSpecified directory for saving the model doesn't exist. Application will end.")


def check_arch(architecture):
    if architecture == 'densenet161' or architecture == 'vgg16':
        return True
    else:
        print(f'Unknown architecture (provided: {architecture}). Please pick from: densenet161 or vgg16.')


parser = argparse.ArgumentParser(description='Train and save an image classification model.')
parser.add_argument('data_dir', help='The directory of images you want the model to be trained on', type=str)
parser.add_argument('--save_dir', help='Directory to save checkpoints', type=str, default='save_directory')
parser.add_argument('--architecture', help='Architecture of the network', default='vgg16', type=str)

args = parser.parse_args()

data_dir  = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir  = data_dir + '/test'





if torch.cuda.is_available():
    print('-'*10, '\nCUDA used:', torch.cuda.memory_allocated(), '\n', '-'*10)
    device = torch.device('cuda')
else:
    print("CUDA was not found on device, using CPU instead.")
    device = torch.device('cpu')

if data_dir:
    if check_save_dir(args.save_dir) and check_arch(args.architecture):
        # transform data for training, testing and validation sets
        # and then load the datasets with ImageFolder
        train_data = test_transformer(train_dir)
        test_data = train_transformer(test_dir)
        val_data = train_transformer(valid_dir)
            
        # define the dataloaders
        trainloader = data_loader(train_data, shuffle=True)
        testloader = data_loader(test_data)
        valloader = data_loader(val_data)
        
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)
        
        model = create_model(args.architecture)
#             model = models.vgg16(pretrained=True)
        model.to(device)

        for param in model.parameters():
            param.requires_grad = False

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

        model.classifier = classifier.to(device)

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
        save_model(args.save_dir)
    
    
    
        
    
