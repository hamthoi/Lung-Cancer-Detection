"""app: A Flower / PyTorch app."""

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, ToTensor, Grayscale, Resize


class Net(nn.Module):
    """Model (simple CNN for 4-class lung cancer detection)"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Update the input size for fc1 after conv/pool layers (see below)
        self.fc1 = nn.Linear(16 * 91 * 117, 120)  # Updated below
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)  # 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def load_data(data_dir: str):
    """Load lung cancer data."""
    pytorch_transforms = Compose(
        [
            Grayscale(num_output_channels=1),  # Convert to greyscale
            Resize((376, 480)),                # Resize to 376x480 (HxW)
            ToTensor(),
            Normalize((0.5,), (0.5,)),         # Single channel mean/std
        ]
    )
    # Use pathlib for robust path handling
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    valid_dir = data_path / "valid"
    test_dir = data_path / "test"

    trainset = ImageFolder(str(train_dir), transform=pytorch_transforms)
    validset = ImageFolder(str(valid_dir), transform=pytorch_transforms)
    testset = ImageFolder(str(test_dir), transform=pytorch_transforms)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    validloader = DataLoader(validset, batch_size=32)
    testloader = DataLoader(testset, batch_size=32)
    return trainloader, validloader, testloader


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
