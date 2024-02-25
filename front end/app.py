from flask import Flask, render_template, request, jsonify, redirect, url_for
from inference import load_model, predict_image
import os
from inference import load_model, predict_image

app = Flask(__name__)

# Load the model, class labels, mean, and std
model_load_path = 'pretrained_mobilenetv2.pth'
model, class_labels, mean, std = load_model(model_load_path)

answers_dict = {}
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

@app.route('/lung_cancer')
def lung_cancer():
    return render_template('lungcancer.html')

@app.route('/submit', methods=['POST'])
def submit():
    global answers_dict
    data = request.json
    answers_dict.update(data)
    print(answers_dict)
    return jsonify({'message': 'Data received successfully'})

@app.route('/classify', methods=['POST'])

def upload_image():
    if 'fileInput' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['fileInput']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Save the uploaded file to the uploads directory
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'temp.jpg'))

    #Classification
    model_load_path = 'pretrained_mobilenetv2.pth'

    # Load the model, class labels, mean, and std
    model, class_labels, mean, std = load_model(model_load_path)

    # Path to the input image
    input_image_path = '/Users/freddiehurdwood/Desktop/Uni Work/ForCancer/uploads/temp.jpg'

    # Perform inference
    predicted_class, probabilities = predict_image(input_image_path, model, class_labels, mean, std)
    print(predicted_class, probabilities)
    os.remove('/Users/freddiehurdwood/Desktop/Uni Work/ForCancer/uploads/temp.jpg')
#classifcation ends
    
    return jsonify({'message': 'Image saved successfully'})


@app.route('/result')
def result():
    predicted_class = request.args.get('predicted_class')
    probabilities = request.args.get('probabilities')
    return render_template('result.html', predicted_class=predicted_class, probabilities=probabilities)

if __name__ == '__main__':
    app.run(debug=True)
