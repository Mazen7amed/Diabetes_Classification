import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

## Loading Data
df = pd.read_csv("Notebook/diabetes_prediction_dataset.csv")

## Drop duplicates
df.drop_duplicates(inplace=True)

df['gender'] = df['gender'].replace('Other', df['gender'].mode()[0])
df['smoking_history'] = df['smoking_history'].replace('No Info', df['smoking_history'].mode()[0])




df['gender'] = df['gender'].replace('Female',0)
df['gender'] = df['gender'].replace('Male', 1)



features = ['smoking_history']

df = pd.get_dummies(data=df,columns=features)

for col in df.columns.to_list():
    if df[col].dtype == bool:
        df[col] = df[col].astype(int)


columns_to_scale = ['age', 'bmi', 'HbA1c_level',
        'blood_glucose_level']

df[columns_to_scale] = RobustScaler().fit_transform(df[columns_to_scale])




     
## Balancing The Data
x = df.drop('diabetes',axis=1)
y = df['diabetes']


oversample = SMOTEENN(enn=EditedNearestNeighbours(sampling_strategy='auto', n_neighbors=5),random_state=42)

x_res, y_res = oversample.fit_resample(x, y)

X_train , X_test , Y_train , Y_test = train_test_split(x_res,y_res, test_size=0.33 , random_state=42)
    
