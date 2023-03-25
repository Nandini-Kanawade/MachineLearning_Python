#Data & Parameter Calulation
import numpy as np
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5,7,10,12,15,20])
y = np.array([0,0.5,0.6,1,2,4,5,5.5,6,7])
plt.scatter(x,y)

#Dividing the above dataSet for Training(70%) and testing(30%)
from sklearn.model_selection import train_test_split
[xtrain, xtest, ytrain, ytest]=train_test_split(x,y)
print(xtrain, ytrain) 

#Parameters for beta1 and beta0 calulations
n = np.size(xtrain)
sum_x =np.sum(xtrain)
sum_y = np.sum(ytrain)
sum_xy = np.sum(xtrain*ytrain)
sum_x2 = np.sum(xtrain**2)
mean_x= np.mean(xtrain)
mean_y=np.mean(ytrain)

#Calulating beta1
nume = (n*sum_xy) - (sum_x*sum_y)
deno = (n*sum_x2) - (sum_x**2)
beta1=nume/deno
#Calculating beta0
beta0 = mean_y -(beta1*mean_x)

print("beta 0 =",beta0)
print("beta 1 =",beta1)

#Calulation by OLS method
#observe that both the above calculated are same
import statsmodels.api as sm
xtrainNew =sm.add_constant(xtrain)
ytrain_pred = sm.OLS(ytrain,xtrainNew).fit()
print(ytrain_pred.params)

#TESTING
# xtest is the dataset on which
# we will be performing the prediction 
#This ytest is expected output from Predicted Y 
print("Expected Output:= ",ytest)
pred_y = np.array([beta0 + (beta1*xtest)])
print("Predicted Output: ",pred_y)

#predicting the output via different method
#observe that both the above and below are same.
xtestNew =sm.add_constant(xtest)
pred2_y = ytrain_pred.predict(xtestNew)
print(pred_y)
print(pred2_y)

#Calculating R2 and RMSE
from sklearn.metrics import r2_score, mean_squared_error
R2=np.abs(r2_score(ytest,pred_y))
print("R2=",R2)
RMSE = np.sqrt(mean_squared_error(ytest,pred_y))
print("RMSE=", RMSE)
