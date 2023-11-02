from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
app = Flask(__name__, static_url_path='/static')


model = joblib.load('Deployment/model.pkl')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from form
    input_data = []
    column_names = ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
       'blood_glucose_level', 'smoking_history_current',
       'smoking_history_ever', 'smoking_history_former',
       'smoking_history_never', 'smoking_history_not current']
    
    for column in column_names :
        input_data.append(float(request.form[column]))

    
    # Convert input data to DataFrame
    input_array = np.array(input_data).reshape(1,12)

    # preprocessing input data
    x_df = pd.DataFrame(input_array, columns=column_names)
    
    columns_to_scale = ['age', 'bmi', 'HbA1c_level',
        'blood_glucose_level']

    x_df[columns_to_scale] = RobustScaler().fit_transform(x_df[columns_to_scale])

    # Make Prediction
    prediction = model.predict(x_df)

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)