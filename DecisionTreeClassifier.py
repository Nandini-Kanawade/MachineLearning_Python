# importing all the necessary libraries and files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import tree

# Read and print the csv file to know the features and target
df = pd.read_csv("/content/User_Data.csv")
print(df.shape)
print("\n")
df.describe

# As the range of EstimatedSalary and Age are too different,
# we need to scale both the features.
#Selecting the features to be scaled
ScalingFeatures = df.iloc[:, 2:4]
print(ScalingFeatures)

# Using the StandardScalar funtion for Scaling 
Scaler = StandardScaler()
Scaled = Scaler.fit_transform(ScalingFeatures)
print(Scaled)

# Now inserting the 2 new columns in the dataSet
# of scaled Age and Salary and Converting the Gender column
# in the form of numeric data.
df.insert(2, "NewAge", Scaled[:,0], True)
df.insert(3, "NewSalary", Scaled[:,1], True)
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
print(df)

# Selecting the features to be Gender, NewAge, NewSalary
# and target to be Purchased column and Splitting the data set
# into training and testing data.
features = df.iloc[:, 1:4]
target = df['Purchased']
[xtrain, xtest, ytrain, ytest] = train_test_split(features,target, test_size = 0.3)

# Using the Decision Tree Classifier
mymodel = DecisionTreeClassifier()
# Training the model 
mymodel.fit(xtrain,ytrain)
# Predicting the values
ypred = mymodel.predict(xtest)
# Getting the Accuracy
confmat = confusion_matrix(ytest, ypred)
print(confmat)
accuracy = metrics.accuracy_score(ytest, ypred)
print("\nAccuracy:= ",accuracy)
# Plotting the Tree
fig = plt.figure(figsize=(25,20))
figure1 = tree.plot_tree(mymodel,filled=True)

# To improve the Accuracy we will Tune the hyperparameters 
# that is change the Criterion to entropy and Prun the tree 
# at depth 3  
mymodel2 = DecisionTreeClassifier(criterion='entropy', max_depth=3)
# Training the model 
mymodel2.fit(xtrain,ytrain)
# Predicting the values
ypred2 = mymodel2.predict(xtest)
# Getting the Accuracy
confmat2 = confusion_matrix(ytest, ypred2)
print(confmat2)
accuracy2 = metrics.accuracy_score(ytest, ypred2)
print("\nAccuracy:= ",accuracy2)
# Plotting the Tree
fig = plt.figure(figsize=(25,20))
figure1 = tree.plot_tree(mymodel2,filled=True)

# changing the Criterion to entropy and Prun the tree at depth 4
mymodel3 = DecisionTreeClassifier(criterion='entropy',max_depth=4)
# Training the model 
mymodel3.fit(xtrain,ytrain)
# Predicting the values
ypred3 = mymodel3.predict(xtest)
# Getting the Accuracy
confmat3 = confusion_matrix(ytest, ypred3)
print(confmat3)
accuracy3 = metrics.accuracy_score(ytest, ypred3)
print("\nAccuracy:= ",accuracy3)
# Plotting the Tree
fig = plt.figure(figsize=(25,20))
figure1 = tree.plot_tree(mymodel3,filled=True)

# changing the Criterion to entropy and Prun the tree at depth 5 
mymodel4 = DecisionTreeClassifier(criterion='gini',max_depth=5)
# Training the model 
mymodel4.fit(xtrain,ytrain)
# Predicting the values
ypred4 = mymodel4.predict(xtest)
# Getting the Accuracy
confmat4 = confusion_matrix(ytest, ypred4)
print(confmat4)
accuracy4 = metrics.accuracy_score(ytest, ypred4)
print("\nAccuracy:= ",accuracy4)
# Plotting the Tree
fig = plt.figure(figsize=(25,20))
figure1 = tree.plot_tree(mymodel4,filled=True)
