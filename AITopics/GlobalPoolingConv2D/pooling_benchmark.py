import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.utils import make_grid
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm.notebook import tqdm


# Device Support
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Running on GPU')
else:
    device = torch.device('cpu')
    print('Running on GPU')

# Training and Validation data
train_set = datasets.FashionMNIST(root='./', download=True,
                                  transform=transforms.ToTensor())
valid_set = datasets.FashionMNIST(root='./', download=True, train=False,
                                  transform=transforms.ToTensor())

# ConvNet with Global Average Pooling


class ConvNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            #  layer 1
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),  # feature map size = (28, 28)
            #  layer 2
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # feature map size = (14, 14)
            #  layer 3
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),  # feature map size = (14, 14)
            #  layer 4
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # feature map size = (7, 7)
            #  layer 5
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),  # feature map size = (7, 7)
            #  layer 6
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # feature map size = (3, 3)
            #  output layer
            nn.Conv2d(32, 10, 1),
            nn.AvgPool2d(3)
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        output = self.network(x)
        output = output.view(-1, 10)
        return torch.sigmoid(output)

# ConvNet with Global Max Pooling


class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            #  layer 1
            nn.Conv2d(1, 8, 3, padding=1),
            nn.ReLU(),  # feature map size = (28, 28)
            #  layer 2
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # feature map size = (14, 14)
            #  layer 3
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),  # feature map size = (14, 14)
            #  layer 4
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # feature map size = (7, 7)
            #  layer 5
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),  # feature map size = (7, 7)
            #  layer 6
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # feature map size = (3, 3)
            #  output layer
            nn.Conv2d(32, 10, 1),
            nn.MaxPool2d(3)
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        output = self.network(x)
        output = output.view(-1, 10)
        return torch.sigmoid(output)


class CNN():
    def __init__(self, network):
        self.network = network.to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=3e-4)

    def train(self, loss_function, epochs, batch_size,
              training_set, validation_set):
        #  creating log
        log_dict = {
            'training_loss_per_batch': [],
            'validation_loss_per_batch': [],
            'training_accuracy_per_epoch': [],
            'validation_accuracy_per_epoch': []
        }

        #  defining weight initialization function
        def init_weights(module):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)

        #  defining accuracy function
        def accuracy(network, dataloader):
            total_correct = 0
            total_instances = 0
            for images, labels in tqdm(dataloader):
                images, labels = images.to(device), labels.to(device)
                predictions = torch.argmax(network(images), dim=1)
                correct_predictions = sum(predictions == labels).item()
                total_correct += correct_predictions
                total_instances += len(images)
            return round(total_correct/total_instances, 3)

        #  initializing network weights
        self.network.apply(init_weights)

        #  creating dataloaders
        train_loader = DataLoader(training_set, batch_size)
        val_loader = DataLoader(validation_set, batch_size)

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            train_losses = []

            #  training
            print('training...')
            for images, labels in tqdm(train_loader):
                #  sending data to device
                images, labels = images.to(device), labels.to(device)
                #  resetting gradients
                self.optimizer.zero_grad()
                #  making predictions
                predictions = self.network(images)
                #  computing loss
                loss = loss_function(predictions, labels)
                log_dict['training_loss_per_batch'].append(loss.item())
                train_losses.append(loss.item())
                #  computing gradients
                loss.backward()
                #  updating weights
                self.optimizer.step()
            with torch.no_grad():
                print('deriving training accuracy...')
                #  computing training accuracy
                train_accuracy = accuracy(self.network, train_loader)
                log_dict['training_accuracy_per_epoch'].append(train_accuracy)

            #  validation
            print('validating...')
            val_losses = []

            with torch.no_grad():
                for images, labels in tqdm(val_loader):
                    #  sending data to device
                    images, labels = images.to(device), labels.to(device)
                    #  making predictions
                    predictions = self.network(images)
                    #  computing loss
                    val_loss = loss_function(predictions, labels)
                    log_dict['validation_loss_per_batch'].append(
                        val_loss.item())
                    val_losses.append(val_loss.item())
                #  computing accuracy
                print('deriving validation accuracy...')
                val_accuracy = accuracy(self.network, val_loader)
                log_dict['validation_accuracy_per_epoch'].append(val_accuracy)

            train_losses = np.array(train_losses).mean()
            val_losses = np.array(val_losses).mean()

            print(f'training_loss: {round(train_losses, 4)}  training_accuracy: ' +
                  f'{train_accuracy}  validation_loss: {round(val_losses, 4)} ' +
                  f'validation_accuracy: {val_accuracy}\n')

        return log_dict

    def predict(self, x):
        return self.network(x)

def visualize_layer(model, dataset, image_idx: int, layer_idx: int):
    dataloader = DataLoader(dataset, 250)
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

    # Derive output from Layer of Intrest
    output = model.network.network[:layer_idx].forward(images[image_idx])
    out_shape = output.shape

    # Classify image
    predicted_class = model.predict(images[image_idx])
    print(f'actual class: {labels[image_idx]}\npredicted class: {torch.argmax(predicted_class)}')

    # Visualising layer
    plt.figure(dpi=150)
    plt.title(f'visualising output')
    plt.imshow(np.transpose(make_grid(output.cpu().view(out_shape[0], 1, 
                                                            out_shape[1], 
                                                            out_shape[2]), 
                                        padding=2, normalize=True), (1,2,0)))


if __name__ == '__main__':
    model1 = CNN(ConvNet1())
    log_dict1 = model1.train(nn.CrossEntropyLoss(), epochs=60, batch_size=64, training_set=train_set, validation_set=valid_set)

    sns.lineplot(y=log_dict1['training_accuracy_per_epoch'], x=range(
        len(log_dict1['training_accuracy_per_epoch'])), label='training')

    sns.lineplot(y=log_dict1['validation_accuracy_per_epoch'], x=range(
        len(log_dict1['validation_accuracy_per_epoch'])), label='validation')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
    
    # Model1
    # training_loss: 1.5325  training_accuracy: 0.878  validation_loss: 1.5471 validation_accuracy: 0.867
    # Model 2
    model2 = CNN(ConvNet2())

    log_dict_2 = model2.train(nn.CrossEntropyLoss(), epochs=60, batch_size=64,
                           training_set=train_set, validation_set=valid_set)
    sns.lineplot(y=log_dict_2['training_accuracy_per_epoch'], x=range(
        len(log_dict_2['training_accuracy_per_epoch'])), label='training')

    sns.lineplot(y=log_dict_2['validation_accuracy_per_epoch'], x=range(
        len(log_dict_2['validation_accuracy_per_epoch'])), label='validation')

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('maxpool_benchmark.png', dpi=1000)
    plt.show()
    """
    0%|          | 0/157 [00:00<?, ?it/s]
    training_loss: 1.5329  training_accuracy: 0.871  validation_loss: 1.5513 validation_accuracy: 0.862
    """
    # How GlobalMax and GlobalAvgPooling works??
    print(model1.network)
    print(model2.network)
    # GlobalAvg
    visualize_layer(model=model1, dataset=valid_set, image_idx=2, layer_idx=16)
    # GlobaMax
    visualize_layer(model=model2, dataset=valid_set, image_idx=2, layer_idx=16)