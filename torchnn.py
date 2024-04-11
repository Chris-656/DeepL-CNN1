# Import dependencies
import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

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

# Get data

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

trainSet = datasets.MNIST(root="train", download=True, train=True, transform=ToTensor())
trainLoader = DataLoader(trainSet, batch_size=128, shuffle=True)
#1,28,28 - classes 0-9


valSet = datasets.MNIST('test', download=True, train=False, transform=transform)
valLoader = DataLoader(valSet, batch_size=64, shuffle=True)


dataiter = iter(trainLoader)
images, labels = dataiter._next_data()

print(images.shape)
print(labels.shape)

figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
plt.show()

# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to('cuda')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training flow
if __name__ == "__main__":
    for epoch in range(10): # train for 10 epochs
        for batch in trainLoader:
            X,y = batch
            X, y = X.to('cuda'), y.to('cuda')
            yhat = clf(X)
            loss = loss_fn(yhat, y)

            # Apply backprop
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"Epoch:{epoch} loss is {loss.item()}")

    with open('model_state.pt', 'wb') as f:
        save(clf.state_dict(), f)

    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    img = Image.open('img_9.jpg')
    img_tensor = ToTensor()(img).unsqueeze(0).to('cuda')

    print(torch.argmax(clf(img_tensor)))