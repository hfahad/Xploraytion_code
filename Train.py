# Import required libraries and modules
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from NeuralNet import NeuralNet
import os


def Train(batch_size, nb_epochs, model_path, in_channels, out_channels, kernal):
    """
    Train the Neural Network on the MNIST dataset

    Parameters:
        batch_size (int): batch size for data loading
        nb_epochs (int): number of training epochs
        model_dir (str): directory to save the model

    Returns:
        None
    """

    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root='data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(
        root='data', train=False, transform=transforms.ToTensor(), download=True)

    # Define dataloaders for the train and test datasets
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the Neural Network
    Net = NeuralNet(in_channels=in_channels,
                    out_channels=out_channels, kernel=kernal)

    # Send the Neural Network to the device and set it in training mode
    Net.to(device)
    Net.train()

    # Define the optimizer and loss function
    optimizer = optim.Adam(params=Net.parameters(), lr=0.003)
    criterion = nn.CrossEntropyLoss()

    # Loop over the number of epochs
    for n in range(nb_epochs):
        correct = 0
        running_loss = 0
        print(f'Epoch: {n+1}/{nb_epochs}')

        # Loop over the training data
        for (data, target) in tqdm(train_loader):

            data, target = data.to(device), target.to(device)

            # Forward pass
            output = Net(data)
            loss = criterion(output, target)

            # Backward pass and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update correct predictions and running loss
            _, pred = torch.max(output.data, 1)
            correct += (pred == target).sum().item()
            running_loss += loss.item()

        # Print training loss and accuracy after each epoch
        print('\nAverage training Loss: {:.4f}, training Accuracy: \
            {}/{} ({:.3f}%)\n'.format(
            loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))

        # Evaluation of the model on the test set
        with torch.no_grad():
            Net.eval()
            loss = 0
            correct = 0

            for data, target in test_loader:
                data = data.unsqueeze(1)
                target = target.squeeze(1)
                data, target = data.to(device), target.to(device)

                output = Net(data)
                loss += F.cross_entropy(output, target, reduction='sum').item()

                _, pred = torch.max(output.data, 1)
                correct += (pred == target).sum().item()

            loss /= len(test_loader.dataset)

            print('Average Val Loss: {:.4f}, Val Accuracy: {}/{} \
                ({:.3f}%)\n'.format(
                loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    Path = model_path + '/model_mnist.pth'
    torch.save(Net.state_dict(), Path)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Add argument for batch size
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=64,
                        help="batch_size")

    # Add argument for number of epochs
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=100,
                        help="number of iterations")

    # Add argument for data folder
    parser.add_argument("--model_path", type=str,
                        dest="model_path", default='./model_path',
                        help="model save folder")

    # Add argument for data folder
    parser.add_argument("--in_channels", type=int,
                        dest="in_channels", default=1,
                        help="1 for GRAY and 3 for RGB")

    # Add argument for data folder
    parser.add_argument("--out_channels", type=int,
                        dest="out_channels", default=10,
                        help="Total numbers of classes")

    # Add argument for data folder
    parser.add_argument("--kernal", type=int,
                        dest="mkernal", default=3,
                        help="convolutional kernal size")

    args = parser.parse_args()
    Train(**vars(args))
