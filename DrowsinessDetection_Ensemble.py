#importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
 
# importing machine learning models for classification
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

 
# importing voting classifier
from sklearn.ensemble import VotingClassifier
 
# loading our dataset into the dataframe-df
df = pd.read_csv(r"C:/Users/5CD034CN3L/Desktop/Jupyter/DriverDrowsinessDetection/eye_dataset_2.csv") #pandas' function to read csv file
df= df.drop(columns = 'Id')

# getting target data from the dataframe
Y = df["Class"]
 
# getting train data from the dataframe
X = df.drop(columns="Class")
 
# Splitting between train data into training and validation dataset
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.30)
 
# initializing all the model objects with default parameters
model_1 = KNeighborsClassifier()
model_2 = XGBClassifier()
model_5=SVC()
model_4 = DecisionTreeClassifier()
model_3 = RandomForestClassifier()
 
# Making the final model using voting classifier
final_model = VotingClassifier(
    estimators=[ ('knn',model_1),('xgb', model_2), ('rf', model_3),('dT',model_4),('svm',model_5)], voting='hard')
 
# training all the model on the train dataset
final_model.fit(x_train, y_train)

print("Accuracy :",final_model.score(x_test,y_test)*100) #accuracy of  the model on test dataset
print()

#importing confusion matrix metric
from sklearn.metrics import confusion_matrix

y_predict = final_model.predict(x_test) 
print("Confusion matrix :")
print(confusion_matrix(y_test, y_predict)) #using confusion matrix as a metric