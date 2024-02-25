from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Handle image classification using your CNN model
    # For demonstration purposes, let's assume the result is 'Cancerous'
    result = 'cancerous'
    return result

@app.route('/result')
def result():
    result = request.args.get('result')  # Get result from query parameter
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
