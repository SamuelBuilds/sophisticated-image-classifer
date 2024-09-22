import torch
from torchvision import models
from torch import nn, optim

def build_model(arch='vgg13', hidden_units=512):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError("Unsupported architecture. Please choose 'vgg13' or 'vgg16'.")

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # feed-forward classifier
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102), 
        nn.LogSoftmax(dim=1)
    )

    return model

def train_model(trainloader, validloader, arch='vgg13', hidden_units=512, learning_rate=0.001, epochs=10, gpu=False):
    model = build_model(arch, hidden_units)

    # Move the model to GPU if available
    device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation loop
        model.eval()
        valid_loss = 0
        accuracy = 0
        
        with torch.no_grad():
            for inputs, labels in validloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                
                # Calculate accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        # Print training and validation statistics
        print(f"Epoch {epoch + 1}/{epochs}.. "
              f"Training loss: {running_loss / len(trainloader):.3f}.. "
              f"Validation loss: {valid_loss / len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy / len(validloader):.3f}")

    return model
