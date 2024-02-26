import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import numpy as np


class TumorClassifier(nn.Module):
        def __init__(self, num_classes):
            super(TumorClassifier, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.classifier = nn.Sequential(
                nn.Linear(32 * 56 * 56, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

def perform_inference(image_path, model_weights_path):
    # Define the transformation for inference
    inference_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the model
    model = TumorClassifier(num_classes=4)  # Assuming TumorClassifier is defined in the same way as in the training script
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    # Load and preprocess the image for inference
    image = Image.open(image_path)
    input_tensor = inference_transform(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Post-process the output probabilities
    probabilities = torch.softmax(output, dim=1)[0]
    predicted_class_index = torch.argmax(probabilities).item()
    predicted_class_prob = probabilities[predicted_class_index].item()

    # Map index to class label (assuming class indices are mapped to class labels in the same order)
    class_labels = ['a glioma', 'a meningioma', 'no tumor', 'a pituitary tumour']  # Replace with your actual class labels

    predicted_class_label = class_labels[predicted_class_index]
    predicted_class_prob = predicted_class_prob * 100
    predicted_class_prob = np.round(predicted_class_prob, 2)


    return predicted_class_label, predicted_class_prob

