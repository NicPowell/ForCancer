import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import pandas as pd

# Define the path to the saved model
model_load_path = 'pretrained_mobilenetv2.pth'

# Load the model and metadata
checkpoint = torch.load(model_load_path)
num_classes = checkpoint['num_classes']
class_labels = checkpoint['class_labels']
mean = checkpoint['mean']
std = checkpoint['std']

# Load the model architecture
pretrained_mobilenetv2 = torchvision.models.mobilenet_v2(pretrained=False)
pretrained_mobilenetv2.classifier[1] = nn.Linear(pretrained_mobilenetv2.last_channel, num_classes)

# Load the model state dictionary
pretrained_mobilenetv2.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
pretrained_mobilenetv2.eval()

# Preprocess the input image
def preprocess_image(image_path):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    # Open the image
    image = Image.open(image_path)
    # Apply transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Define a function to perform inference
def predict_image(image_path):
    # Preprocess the image
    input_image = preprocess_image(image_path)
    
    # Perform inference
    with torch.no_grad():
        # Forward pass
        logits = pretrained_mobilenetv2(input_image)
        # Get predicted class probabilities
        probabilities = torch.softmax(logits, dim=1)[0]
        # Get the predicted class index
        predicted_class_index = torch.argmax(probabilities).item()
    
    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class_index]
    
    return predicted_class_label, probabilities

# Example usage
if __name__ == "__main__":
    # Path to the input image
    input_image_path = '/Users/freddiehurdwood/Desktop/Uni Work/ForCancer/freddie/data/images/ISIC_0034314.jpg'  # Replace with the path to your input image
    
    # Perform inference
    predicted_class, probabilities = predict_image(input_image_path)
    
    # Output the prediction
    print("Predicted class:", predicted_class)
    print("Class probability:", "{:.2f}%".format(probabilities.max() * 100))
