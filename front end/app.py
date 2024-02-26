from flask import Flask, render_template, request, jsonify, redirect, url_for
from inference import load_model, predict_image
import os
from inference_brain import perform_inference
from NN import predict

app = Flask(__name__)

# Load the model, class labels, mean, and std
model_load_path = 'pretrained_mobilenetv2.pth'
model, class_labels, mean, std = load_model(model_load_path)


UPLOAD_FOLDER = 'uploads'  # Directory to store uploaded files
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/skin_cancer')
def skin_cancer():
    return render_template('skincancer.html')

@app.route('/brain_cancer')
def brain_cancer():
    return render_template('brain_cancer.html')

@app.route('/lung_cancer')
def lung_cancer():
    return render_template('lungcancer.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Handle form submission
    data = request.json
    predictions = predict(data)

    print(predictions)
    return jsonify({'message': 'Data received successfully'})

@app.route('/classify', methods=['POST'])
def upload_image():
    if 'fileInput' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['fileInput']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file to the uploads directory
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'temp.jpg'))

    # Classification
    # Load the model, class labels, mean, and std
    model, class_labels, mean, std = load_model(model_load_path)

    # Path to the input image
    input_image_path = 'uploads/temp.jpg'

    # Perform inference
    predicted_class, probabilities = predict_image(input_image_path, model, class_labels, mean, std)
    probabilities = float(probabilities)
    print(predicted_class, probabilities)
    os.remove('uploads/temp.jpg')

    # Render result template with predicted class and probabilities
    return jsonify({'predicted_class': predicted_class, 'probabilities': probabilities})


@app.route('/classify_brain', methods=['POST'])
def upload_image_brain():
    if 'fileInput' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['fileInput']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file to the uploads directory
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'temp.jpg'))




    # Path to the input image
    input_image_path = 'uploads/temp.jpg'

    # Perform inference
    predicted_class, probabilities = perform_inference(input_image_path, '/Users/freddiehurdwood/Desktop/Uni Work/ForCancer/brain_best_model2.pth')
    probabilities = float(probabilities)
    print(predicted_class, probabilities)
    os.remove('uploads/temp.jpg')

    # Render result template with predicted class and probabilities
    return jsonify({'predicted_class': predicted_class, 'probabilities': probabilities})


@app.route('/result_skin')
def result():
    predicted_class = request.args.get('predicted_class')
    probabilities = request.args.get('probabilities')
    print('hello',predicted_class)
    return render_template('result_skin.html', predicted_class=predicted_class, probabilities=probabilities)

@app.route('/result_brain')
def result_brain():
    predicted_class = request.args.get('predicted_class')
    probabilities = request.args.get('probabilities')
    print('hello',predicted_class)
    return render_template('result_brain.html', predicted_class=predicted_class, probabilities=probabilities)



if __name__ == '__main__':
    app.run(debug=True)
