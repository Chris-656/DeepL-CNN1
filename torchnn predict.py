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
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        return self.model(x)

# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":

    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    img = Image.open('img_9.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')

    print(torch.argmax(clf(img_tensor)))


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