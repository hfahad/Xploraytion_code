# Importing required libraries
from NeuralNet import NeuralNet
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


def Evaluation(model_path):
    """
    Function to evaluate the model performance on the test data

    Parameters:
        model_path: string, path to the trained model's

    Returns:
        None
    """
    # Set the device to use (either CPU or GPU)
    device = ("cuda" if torch.cuda().is_available() else "cpu")

    # Load the trained NeuralNet model
    Net = NeuralNet(1, 10, 3).to(device)
    Net.load_state_dict(torch.load(os.path.join(
        model_path, "model_mnist.pth"), map_location='cpu'))

    # Load the test dataset
    test_dataset = datasets.MNIST(
        root='data', train=False, transform=transforms.ToTensor(), download=True)

    # Define dataloader for the test dataset
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False)

    # Set the model in evaluation mode
    with torch.no_grad():
        Net.eval()

        # Initialize the loss and correct count
        loss = 0
        correct = 0
        y_pred = []
        y_true = []

        # Iterate over the test dataloader
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = Net(data)
            loss += F.cross_entropy(output, target, reduction='sum').item()

            # Get the predicted class index
            _, pred = torch.max(output.data, 1)
            correct += (pred == target).sum().item()

            y_pred.extend(pred.cpu().numpy())
            y_true.extend(target.numpy())

        # Compute average loss and accuracy
        loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)

        # Print the average loss and accuracy
        print('Average Val Loss: {:.4f}, Val Accuracy: {}/{} ({:.3f}%)\n'.format(
            loss, correct, len(test_loader.dataset), accuracy))

        # Save the predicted and true class labels
        np.savetxt('/results/y_pred.txt', y_pred, fmt='%d')
        np.savetxt('/results/y_true.txt', y_true, fmt='%d')

        # Plot the confusion matrix
        Confusion_matrix(y_true, y_pred)


def Confusion_matrix(y_true, y_pred):
    """
    Function to plot the confusion matrix

    Parameters:
        y_true: numpy array, y_pred: numpy array
    """
    # Calculate confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)

    # Define class names
    class_names = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # Create a dataframe from confusion matrix
    dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)

    # Plot the confusion matrix using Seaborn heatmap
    sns.heatmap(dataframe, annot=True, cbar=None, cmap='YlGnBu', fmt='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')

    # Save the confusion matrix plot to file
    plt.savefig('/results/cm_matrix.jpg', bbox_inches='tight')


if __name__ == "__main__":

    Evaluation('./model_path')
