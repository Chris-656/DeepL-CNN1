# Import dependencies
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt



labels_map = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
}
# Get data
trainingData = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
trainLoader = DataLoader(trainingData, 128, shuffle=True)
#1,28,28 - classes 0-9

# Image Classifier Neural Network

# Training flow
if __name__ == "__main__":



    dataiter = iter(trainLoader)
    images, labels = dataiter._next_data()

    print(images.shape)
    print(labels.shape)

    figure = plt.figure(figsize=(10,10))
    figure.tight_layout
    #figure.tight_layout(pad=8.0)

    cols, rows = 10, 10
    for i in range(1, cols * rows + 1):
        sample_idx = int(torch.randint(len(trainingData), size=(1,)).item())
        img, label = trainingData[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray_r")
        plt.title(labels_map[label],y=0.8)
    plt.show()