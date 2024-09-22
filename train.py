import argparse
import torch
from utils import load_data, process_data
from model import train_model

def main():
    parser = argparse.ArgumentParser(description="Train a neural network.")
    parser.add_argument('data_directory', type=str, help='Path to the data directory')
    parser.add_argument('--save_dir', type=str, default='.', help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg13', help='Model architecture (default: vgg13)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (default: 10)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

    args = parser.parse_args()
    
    # Load and process data
    trainloader, validloader, testloader = load_data(args.data_directory)

    # Train the model
    model = train_model(trainloader, validloader, arch=args.arch, 
                        hidden_units=args.hidden_units, 
                        learning_rate=args.learning_rate, 
                        epochs=args.epochs, 
                        gpu=args.gpu)
    
    # Save the checkpoint
    torch.save(model.state_dict(), f"{args.save_dir}/checkpoint.pth")

if __name__ == "__main__":
    main()
