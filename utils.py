import torch
from torch import nn, optim
import numpy as np
from torchvision import datasets, transforms, models
from PIL import Image
import json

def load_data(data_directory):
    #data transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_data = datasets.ImageFolder(data_directory + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_directory + '/valid', transform=valid_transforms)
    test_data = datasets.ImageFolder(data_directory + '/test', transform=test_transforms)

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32,shuffle=False)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32,shuffle=False)

    return trainloader, validloader, testloader

def process_data(image_path):
    '''Scales, crops, and normalizes a PIL image for a PyTorch model,
       returns a Numpy array suitable for input into the model'''
    
    # Load the image
    image = Image.open(image_path)
    
    # Resize the image where the shortest side is 256 pixels, maintaining aspect ratio
    if image.width > image.height:
        image.thumbnail((256, image.height))
    else:
        image.thumbnail((image.width, 256))
    
    # Crop the center of the image to a 224x224 square
    left = (image.width - 224) / 2
    top = (image.height - 224) / 2
    right = (image.width + 224) / 2
    bottom = (image.height + 224) / 2
    image = image.crop((left, top, right, bottom))
    
    # Convert image to numpy array
    np_image = np.array(image) / 255.0  # Scale to range [0, 1]
    
    # Normalize the image using ImageNet's means and standard deviations
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions to match PyTorch's expectations (C x H x W)
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image
    

def load_checkpoint(filepath):
    
    checkpoint = torch.load(filepath)
    
    architecture = checkpoint.get('architecture', 'vgg16')
    
    # Initialize the model
    if architecture == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        raise ValueError("Unsupported architecture: {}".format(architecture))
    
    for param in model.parameters():
        param.requires_grad = False
    
    # Rebuild the classifier
    model.classifier = nn.Sequential(
        nn.Linear(checkpoint['input_size'], 4096),  
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(1024, checkpoint['output_size']),
        nn.LogSoftmax(dim=1)
    )
    
    # Load the saved state dict for the model
    model.load_state_dict(checkpoint['state_dict'])
    
    # Load class_to_idx
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Rebuild the optimizer
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    return model

def predict(model,image_path, topk=5, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Process the image
    image = process_data(image_path)
    
    # Convert image
    image_tensor = torch.from_numpy(image).unsqueeze(0)
    
    image_tensor = image_tensor.float()
    
    # Move the model to GPU if available
    device = torch.device("cuda" if (gpu and torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    image_tensor = image_tensor.to(device)

    model.eval()
    
    # Disable gradient calculation
    with torch.no_grad():
        # Forward 
        output = model.forward(image_tensor)
    
    # Convert the output probabilities
    probabilities = torch.exp(output)
    
    # Get the top K probabilities and corresponding class indices
    top_probs, top_indices = probabilities.topk(topk)
    
    # Convert to CPU and numpy for further processing
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    # Convert the indices 
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_indices]
    return top_probs, top_classes