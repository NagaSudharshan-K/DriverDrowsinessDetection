import pandas as pd
import numpy as np

#reading csv file using pandas
df = pd.read_csv(r'C:/Users/5CD034CN3L/Desktop/Jupyter/eye_dataset_2.csv')

#dropping unnecessary columns from the dataframe
df = df.drop( columns = 'Id')

#importing random forest model from scikit_learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split #used for splitting dataset

#splitting the dataset into test and  train sets
X = df.drop(columns='Class')
Y= df['Class']
x_train, x_test, y_train, y_test = train_test_split(X , Y , test_size=0.30)

#RANDOM_FOREST
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

#Accuracy score for random forest classification
print("Accuracy for Random forest model:",rf.score(x_test,y_test)*100)
print()

#importing metrics from sklearn.metrics
from sklearn.metrics import confusion_matrix

y_predict = rf.predict(x_test)
print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_predict)) # confusion matrix
print()

#gradientBoosting
from sklearn.ensemble import GradientBoostingClassifier

GB_classifier=GradientBoostingClassifier()
x_train, x_test, y_train, y_test = train_test_split(X , Y , test_size=0.30)
GB_classifier.fit(x_train,y_train)
print("Accuracy for Gradient Boosting Classifier:",GB_classifier.score(x_test,y_test)*100)
print()

#XGBoost
from xgboost import XGBClassifier

xgb_classifier = XGBClassifier()
x_train, x_test, y_train, y_test = train_test_split(X , Y , test_size=0.30)
xgb_classifier.fit(x_train,y_train)
print("Accuracy for XGBoost Classifier:",xgb_classifier.score(x_test,y_test)*100)