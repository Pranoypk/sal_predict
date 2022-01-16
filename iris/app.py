#!/usr/bin/env python
# coding: utf-8

# In[ ]:

from keras.models import load_model
import numpy as np
import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)
model = load_model("Iris.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [ float(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    print(int_features)
    prediction = np.argmax(model.predict([int_features]))
    
    output = prediction
    
    return render_template('index.html', prediction_text= "Employee Salary should be $ {}".format(output))

if __name__ == '__main__':
    app.run(debug=True)

