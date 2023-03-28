from flask import Flask, request, render_template, url_for
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/select_model', methods=['POST'])
def select_model():
    model_type = request.form['model_type']
    if model_type == 'linear':
        return render_template('linear_input.html')
    elif model_type == 'logistic':
        return render_template('iris_input.html')
    else:
        return "Invalid model type selected!"

@app.route('/linear_predict', methods=['POST'])
def linear_predict():
    # Load the saved model
    with open('lir_model.pickle', 'rb') as f:
        model = pickle.load(f)

    # Get the input data from the form
    height = float(request.form['height'])

    # Make a prediction
    prediction = model.predict([[height]])[0]

    # Return the prediction
    return render_template('linear_result.html', prediction=prediction)

@app.route('/iris_predict', methods=['POST'])
def iris_predict():
    # Load the saved model
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)

    # Get the input data from the form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Make a prediction
    prediction = model.predict(input_data)[0]

    # Return the prediction
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='localhost',port=8000,debug=True)
