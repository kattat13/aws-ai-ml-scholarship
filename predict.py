""" 2. Predict

    Predict flower name from an image with predict.py along with the probability of that name. 
    That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

    Basic usage: python predict.py /path/to/image checkpoint
    Options: 
    * Return top KK most likely classes: python predict.py input checkpoint --top_k 3 
    * Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json 
    * Use GPU for inference: python predict.py input checkpoint --gpu """


import argparse
import json
import numpy as np
import torch
from collections import OrderedDict
from PIL import Image
from torch import nn
from torchvision import models, transforms


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    im_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    out_image = im_transform(im)
    return out_image


def predict(image_path, model, gpu, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.to(device)
    model.eval()
    
    im = process_image(image_path)
    im = im.unsqueeze_(0)
    im = im.float()
    im.to(device)
    
    if gpu:
        with torch.no_grad():
            output = model.forward(im.cuda())
    else:
        with torch.no_grad():
            output = model.forward(im)
    
    top_prob, top_labels = torch.topk(output, topk)
    top_prob = top_prob.exp()
        
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    
    for label in top_labels.cpu().numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])
        
    return top_prob.cpu().numpy()[0], mapped_classes


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    if checkpoint['arch'] == 'vgg16':
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
    elif checkpoint['arch'] == 'densenet161':
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(2208, 2048)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(0.2)),
            ('fc2', nn.Linear(2048, 512)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(0.2)),
            ('fc3', nn.Linear(512, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))
    
    model.classifier = classifier   
    
    return model


# python predict.py flowers/test/13/image_05745.jpg save_directory/saved_model.pth


parser = argparse.ArgumentParser(description='Train and save an image classification model.')
parser.add_argument('image_path', help='Path to the input image', default='flowers/test/13/image_05744.jpg')
parser.add_argument('checkpoint', help='Checkpoint of the trained model', default='save_directory/saved_model.pth')

args = parser.parse_args()

check_file = args.image_path
checkpoint = args.checkpoint

# loading the model from the checkpoint
model = load_checkpoint(checkpoint)

img = process_image(check_file)

probs, classes = predict(check_file, model, args.gpu, args.top_k)

class_names = [cat_to_name[item] for item in classes]

print('#\tClass name\tProbability')
print('-'*100)
for i in range(len(probs)):
    print("{}".format(i+1),
            "\t{}".format(class_names[i].title()),
            "\t{:.3f}% ".format(probs[i]*100),
            )