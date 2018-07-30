#Support Vector Regression
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
#Importing dataset
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values
#Feature scaling
from sklearn.preprocessing import StandardScaler
sx=StandardScaler()
sy=StandardScaler()
x=sx.fit_transform(x)
y=sy.fit_transform(y)
#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)
yPred=regressor.predict(6.5)
plot.scatter(x,y,color='red')
plot.plot(x,regressor.predict(x),color='blue')
plot.title('SVR Model')
plot.xlabel('Position')
plot.ylabel('Salary')
plot.show()