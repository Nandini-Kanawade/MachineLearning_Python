#'pandas' library for working with data sheets
import pandas as pd

#'matplotlib' library for working with data visulization methods (plots)
import matplotlib.pyplot as plt

#Reading the .csv file
dataFrame = pd.read_csv("/content/diabetes2 (2).csv")

#Dispalying the Data FRame
dataFrame.info()
dataFrame.describe()

#For seeing first 5 samples of the dataSheet
dataFrame.head()

#Defining training and testing data
from sklearn.model_selection import train_test_split
x=dataFrame.iloc[:,0:8]
y=dataFrame.iloc[:,8]

[featureTrain, featureTest, targetTrain, targetTest]= train_test_split(x,y,test_size=0.3)
print(featureTrain, targetTrain) 

#Feature 1:PREGNANACY
#choosing the feature
#1.Pregnancies
import statsmodels.api as sm

training_feature1 = sm.add_constant(featureTrain['Pregnancies'])
testing_feature1 = sm.add_constant(featureTest['Pregnancies'])
print(training_feature1)

from sklearn.linear_model import LogisticRegression
LR_model = LogisticRegression()
LR_model.fit(training_feature1,targetTrain )
Prediction_1 = LR_model.predict(testing_feature1)
LR_model.score(testing_feature1, targetTest)

from sklearn.metrics import confusion_matrix
C_matrix = confusion_matrix(targetTest, Prediction_1)
print(C_matrix)

from sklearn.metrics import classification_report
print(classification_report(targetTest, Prediction_1))

#Feature 2 BLOOD PRESSURE
#choosing the feature
#2.BloodPressure

training_feature1 = sm.add_constant(featureTrain['BloodPressure'])
testing_feature1 = sm.add_constant(featureTest['BloodPressure'])
print(training_feature1)
LR_model = LogisticRegression()
LR_model.fit(training_feature1,targetTrain )
Prediction_1 = LR_model.predict(testing_feature1)
LR_model.score(testing_feature1, targetTest)

C_matrix = confusion_matrix(targetTest, Prediction_1)
print(C_matrix)
print(classification_report(targetTest, Prediction_1))

#Feature 3:INSULIN
#choosing the feature
#2.BloodPressure

training_feature1 = sm.add_constant(featureTrain['Insulin'])
testing_feature1 = sm.add_constant(featureTest['Insulin'])
print(training_feature1)
LR_model = LogisticRegression()
LR_model.fit(training_feature1,targetTrain )
Prediction_1 = LR_model.predict(testing_feature1)
LR_model.score(testing_feature1, targetTest)

C_matrix = confusion_matrix(targetTest, Prediction_1)
print(C_matrix)

print(classification_report(targetTest, Prediction_1))
