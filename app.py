#Import library yang akan digunakan
from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from array import array
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


app = Flask(__name__)

#Load Modelnya
pipeline = joblib.load('Model/pipeline.pkl')




@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/event')
def event():
     return render_template("event.html")

@app.route('/predict', methods=['POST'])
def predict():
    
    id_val = int(request.form['id'])
    mssubclass = int(request.form['mssubclass'])
    lotarea = float(request.form['lotarea'])
    yearbuilt = int(request.form['yearbuilt'])
    salecondition = request.form['salecondition']

    # Gabungkan semua fitur menjadi DataFrame pandas
    input_data = pd.DataFrame([[id_val, mssubclass, lotarea, yearbuilt, salecondition]],
                              columns=['Id', 'MSSubClass', 'LotArea', 'YearBuilt', 'SaleCondition'])

    # Lakukan prediksi
    prediction = pipeline.predict(input_data)
    output = prediction[0]

    # Tampilkan hasil prediksi ke pengguna
    return render_template('event.html', prediction_text="{}".format(output))



if __name__ == '__main__':
    app.run(debug=True)