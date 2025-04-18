import numpy as np
from flask import Flask, render_template,request
import math
import pickle

app = Flask(__name__)

model2 = pickle.load(open(r'D:\Mukesh\uber-prediction\venv\model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(i) for i in request.form.values()]

    final_features = np.array(int_features).reshape(1,-1)

    prediction = model2.predict(final_features)
    output = round(prediction[0],2)
    return render_template('index.html',predict_text = "Number of weekly rides {}".format(math.floor(output)))

if __name__ =='__main__':
    app.run(debug=True)


