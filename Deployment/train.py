import pandas as pd 
from sklearn.metrics import  classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib
from preprocessing import *


model = RandomForestClassifier(max_depth=20,n_estimators=20)
model.fit(X_train,Y_train)



## Classification Report
Y_pred=model.predict(X_test)
matrix = classification_report(Y_test,Y_pred )
print(matrix)





## Save Model
joblib.dump(model, 'Deployment/model.pkl')


