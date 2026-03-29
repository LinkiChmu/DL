import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
from torchinfo import summary

BATCH_SIZE = 32
LR = 0.001
NUM_EPOCHS = 10

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.2860,), std=(0.3530,))
])
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transformer, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transformer, download=True)

# Визуализация одного изображения
labels = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]
image, label = test_dataset[1]
image[0].shape  # torch.Size([28, 28])
plt.imshow(image.squeeze(), cmap='gray')
plt.title(labels[label])
plt.show()


train_iter = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_iter = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 28x28
    nn.ReLU(),
    nn.MaxPool2d(2),  # 14x14

    nn.Conv2d(6, 16, kernel_size=5),  # 10x10
    nn.ReLU(),
    nn.MaxPool2d(2),  # 5х5

    nn.Conv2d(16, 32, kernel_size=3), # 3x3
    nn.ReLU(),

    nn.Flatten(),
    nn.BatchNorm1d(288),
    nn.Linear(288, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
model = model.to(device)
summary(model, input_size=(1, 1, 28, 28), device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def train(model, optimizer, device, n_epochs=5):
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        train_loss, train_acc, train_samples = 0., 0., 0
        test_loss, test_acc, test_samples = 0., 0., 0
        start = time.time()

        model.train()
        for images, labels in train_iter:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            l = criterion(logits, labels)
            l.backward()
            optimizer.step()

            train_loss += l.item() * images.size(0)
            train_acc += (logits.argmax(axis=1) == labels).sum().item()
            train_samples += images.size(0)

        # валидация
        model.eval()            # отключаются слои регуляризации
        with torch.no_grad():   # отключаются градиенты
            for images, labels in test_iter:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                l = criterion(logits, labels)

                test_loss += l.item() * images.size(0)
                test_acc += (logits.argmax(dim=1) == labels).sum().item()
                test_samples += len(images)

        print(f"epoch: {epoch}, taken: {(time.time() - start):.3f}, "
              f"train_loss: {train_loss / train_samples}\ttrain_acc: {train_acc / train_samples:.4f}\t\t"
              f"test_loss: {test_loss / test_samples}\ttest_acc: {test_acc / test_samples}")


train(model, optimizer, n_epochs=NUM_EPOCHS, device=device)