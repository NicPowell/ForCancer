from flask import Flask, render_template
from flask import Flask, request, jsonify
#from NN import predict
app = Flask(__name__)
answers_dict = {}
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
    # Process the data (e.g., store it in a database, perform analysis)
    answers_dict.update(data)  # Update the answers dictionary with the received data
    print(answers_dict)  # Print the updated dictionary (optional)
   # label = predict(answers_dict)
    return jsonify({'message': 'Data received successfully'})
if __name__ == '__main__':
    app.run(debug=True)
