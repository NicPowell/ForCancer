from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/skin_cancer')
def skin_cancer():
    return render_template('skincancer.html')

@app.route('/lung_cancer')
def lung_cancer():
    return render_template('lungcancer.html')

if __name__ == '__main__':
    app.run(debug=True)
