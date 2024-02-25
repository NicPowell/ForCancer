import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd

torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define your custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, file_extension='.jpg'):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.file_extension = file_extension

        # Extract unique classes from the 'dx' column in the CSV file
        self.classes = self.data_frame['dx'].unique()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, "images", f"{self.data_frame.iloc[idx, 0]}{self.file_extension}")
        image = Image.open(img_name)
        label = self.data_frame.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# Define your data transforms
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Path to your CSV file and data directory
csv_file = '/Users/freddiehurdwood/Desktop/Uni Work/ForCancer/freddie/data/good_metadata.csv'
root_dir = '/Users/freddiehurdwood/Desktop/Uni Work/ForCancer/freddie/data'

# Create custom dataset instance
custom_dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=data_transform)

# Split dataset into train, validation, and test sets
train_size = int(0.7 * len(custom_dataset))
val_size = int(0.1 * len(custom_dataset))
test_size = len(custom_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=40)
test_loader = DataLoader(test_dataset, batch_size=40, shuffle=False)

print("Training set size:", len(train_loader.dataset))
print("Validation set size:", len(val_loader.dataset))
print("Test set size:", len(test_loader.dataset))

# Load a pretrained MobileNetV2 model
pretrained_mobilenetv2 = torchvision.models.mobilenet_v2(pretrained=True)
for param in pretrained_mobilenetv2.parameters():
    param.requires_grad = False

# Modify the fully connected layer to match the number of classes in your dataset
num_classes = len(custom_dataset.classes)
pretrained_mobilenetv2.classifier[1] = nn.Linear(pretrained_mobilenetv2.last_channel, num_classes)

# Move the model to the appropriate device
pretrained_mobilenetv2.to(device)

# Set the optimizer
opt = torch.optim.Adam(pretrained_mobilenetv2.parameters(), lr=0.001)

# Define train and test functions
def train(model):
    model.train()
    i = 1
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        opt.zero_grad()
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        print(i, loss)
        i+= 1
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
for epoch in range(5    ):  # You can adjust the number of epochs as needed
    train(pretrained_mobilenetv2)
    test(pretrained_mobilenetv2, epoch)


model_save_path = 'pretrained_mobilenetv2.pth'

# Save the model state dictionary along with any necessary metadata
torch.save({
    'model_state_dict': pretrained_mobilenetv2.state_dict(),
    'num_classes': num_classes,
    'class_labels': custom_dataset.classes,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}, model_save_path)

print("Model saved successfully!")