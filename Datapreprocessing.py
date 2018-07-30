#DATA PREPROCESSING FOR DATA.CSV FILE
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
#Importing the data set with pandas and taking the necessary variables
dataset=pd.read_csv('Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values
#Handling missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])
#Encoding categorical data and removing relational weights among them
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX=LabelEncoder()
x[:,0]=labelEncoderX.fit_transform(x[:,0])
hotEncoder=OneHotEncoder(categorical_features=[0])
x=hotEncoder.fit_transform(x).toarray()
labelEncoderY=LabelEncoder()
y=labelEncoderY.fit_transform(y)
#Splitting dataset into training and test set
from sklearn.cross_validation import train_test_split
xTrain,xTest,yTrain,yTest= train_test_split(x,y,test_size=0.2,random_state=10)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
scX=StandardScaler()
xTrain=scX.fit_transform(xTrain)
xTest=scX.fit_transform(xTest)