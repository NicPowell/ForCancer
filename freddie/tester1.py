from inference import load_model, predict_image

# Define the path to the saved model
model_load_path = 'pretrained_mobilenetv2.pth'

# Load the model, class labels, mean, and std
model, class_labels, mean, std = load_model(model_load_path)

# Path to the input image
input_image_path = '/Users/freddiehurdwood/Desktop/Uni Work/ForCancer/freddie/data/images/ISIC_0034314.jpg'

# Perform inference
predicted_class, probabilities = predict_image(input_image_path, model, class_labels, mean, std)

# Output the prediction
print("Predicted class:", predicted_class)
print("Class probability:", "{:.2f}%".format(probabilities.max() * 100))