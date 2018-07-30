#Simple Linear Regression
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
#Importing the data set with pandas and taking the necessary variables
dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,1].values
#Splitting dataset into training and test set
from sklearn.cross_validation import train_test_split
xTrain,xTest,yTrain,yTest= train_test_split(x,y,test_size=0.2,random_state=10)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
scX=StandardScaler()
xTrain=scX.fit_transform(xTrain)
xTest=scX.fit_transform(xTest)
#Fitting Simple Linear Regression model to training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xTrain,yTrain)
#Predicting results from test set
yPred=regressor.predict(xTest)
#Visualizing the results of training set
plot.scatter(xTrain,yTrain,color='red')
plot.plot(xTrain,regressor.predict(xTrain),color='blue')
plot.title('Salary vs Experience(Training Set)')
plot.xlabel('Yrs of Experience')
plot.ylabel('Salary')
plot.show()
#Visualizing the results of test set
plot.scatter(xTest,yTest,color='red')
plot.plot(xTrain,regressor.predict(xTrain),color='blue')
plot.title('Salary vs Experience(Test Set)')
plot.xlabel('Yrs of Experience')
plot.ylabel('Salary')