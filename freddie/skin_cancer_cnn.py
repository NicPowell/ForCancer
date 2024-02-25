import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

torch.manual_seed(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet models usually require this size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization for ResNet
])

photos_path = "photos"

photos_dataset = ImageFolder(root=photos_path, transform=data_transform)

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(photos_dataset, [0.7, 0.1, 0.2])

train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=100)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print("Training set size:", len(train_loader.dataset))
print("Validation set size:", len(val_loader.dataset))
print("Test set size:", len(test_loader.dataset))

# Load a pre-trained ResNet model
pretrained_resnet = torchvision.models.resnet18(pretrained=True)

# Freeze all the parameters in the pre-trained model
for param in pretrained_resnet.parameters():
    param.requires_grad = False

# Modify the fully connected layer to match the number of classes in your dataset
num_classes = len(photos_dataset.classes)
pretrained_resnet.fc = nn.Linear(pretrained_resnet.fc.in_features, num_classes)

# Move the model to the appropriate device
pretrained_resnet.to(device)

# Set the optimizer
opt = torch.optim.Adam(pretrained_resnet.parameters(), lr=0.001)

# Define train and test functions (similar to your original code)

def train(model):
    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        opt.zero_grad()
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        opt.step()

def test(model, epoch):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            predicted = torch.argmax(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f'Test accuracy after {epoch + 1} epochs: {100 * correct / total} %')

# Train and test the model
for epoch in range(5):  # You can adjust the number of epochs as needed
    train(pretrained_resnet)
    test(pretrained_resnet, epoch)




