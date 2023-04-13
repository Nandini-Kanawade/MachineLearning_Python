import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, classification_report

df = pd.read_csv("Placement_Data_Full_Class.csv")
print(df.shape)
print("\n")
df.describe

features = df.iloc[:, [2,4,7,10,12]]
target = df['status']

[xtrain, xtest, ytrain, ytest] = train_test_split(features,target, test_size = 0.3)

model1 = GaussianNB()
model1.fit(xtrain, ytrain)
prediction = model1.predict(xtest)

confmat = confusion_matrix(ytest, prediction)
print(confmat)

report = classification_report(ytest, prediction)
print(report)
