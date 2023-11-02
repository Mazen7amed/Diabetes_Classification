import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler


column_names = ['gender', 'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
       'blood_glucose_level', 'smoking_history_current',
       'smoking_history_ever', 'smoking_history_former',
       'smoking_history_never', 'smoking_history_not current']

x = np.array([1,40,1,1,33,8.5,190,1,0,0,0,0]).reshape(1,12)
x_df = pd.DataFrame(x, columns=column_names)
columns_to_scale = ['age', 'bmi', 'HbA1c_level',
        'blood_glucose_level']

x_df[columns_to_scale] = RobustScaler().fit_transform(x_df[columns_to_scale])

model = joblib.load('Deployment/model.pkl')
print(model.predict(x_df))