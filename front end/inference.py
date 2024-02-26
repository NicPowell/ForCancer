import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

def load_model(model_load_path):
    # Load the model and metadata
    checkpoint = torch.load(model_load_path)
    num_classes = checkpoint['num_classes']
    class_labels = checkpoint['class_labels']
    mean = checkpoint['mean']
    std = checkpoint['std']

    # Load the model architecture
    model = torchvision.models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, class_labels, mean, std

def preprocess_image(image_path, mean, std):
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

def predict_image(image_path, model, class_labels, mean, std):
    # Preprocess the image
    input_image = preprocess_image(image_path, mean, std)
    
    # Perform inference
    with torch.no_grad():
        # Forward pass
        logits = model(input_image)
        # Get predicted class probabilities
        probabilities = torch.softmax(logits, dim=1)[0]
        # Get the predicted class index
        predicted_class_index = torch.argmax(probabilities).item()
    
    # Get the predicted class label
    predicted_class_label = class_labels[predicted_class_index]
    class_labels = {
       
        0: 'Actinic keratoses and intraepithelial carcinoma / Bowens disease',
        1: 'basal cell carcinoma',
        2: 'benign keratosis-like lesion',
        3: 'dermatofibroma',
        4: 'melanoma',
        5: 'melanocytic nevi',
        6: 'vascular lesion'
        }

    predicted_class_label = class_labels.get(predicted_class_label, "Unknown")
    top_prob = probabilities[predicted_class_index]
    top_prob = top_prob.item()
    rounded_prob = round(top_prob * 100, 2)
    
    return predicted_class_label, rounded_prob
