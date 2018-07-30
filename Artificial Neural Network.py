#Artificial Neural Network

#Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
#importing dataset
dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values
#Encoding categorical data and independent variable
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoderX=LabelEncoder()
x[:,1]=labelEncoderX.fit_transform(x[:,1])
x[:,2]=labelEncoderX.fit_transform(x[:,2])
hotEncoder=OneHotEncoder(categorical_features=[1])
x=hotEncoder.fit_transform(x).toarray()
x=x[:,1:]
#Splitting dataset into training and test set
from sklearn.model_selection import train_test_split as tts
xTrain,xTest,yTrain,yTest=tts(x,y,test_size=0.2,random_state=0)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xTrain=sc.fit_transform(xTrain)
xTest=sc.transform(xTest)
#Building ANN by importing Keras library
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
#Initializing ANN
classifier=Sequential()
#Adding input layer and first hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
classifier.add(Dropout(p=0.1))
#Adding more hidden layers
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
classifier.add(Dropout(p=0.1))
#Adding output layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
#Compiling the ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Fitting ANN to training set
classifier.fit(xTrain,yTrain,epochs=100,batch_size=10)
#Predicting the test set results
yPred=classifier.predict(xTest)
yPred=(yPred>0.5)
#Predicting for new input
yNew=classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
yNew=(yNew>0.5)
#Creating confusion matrix to find TP,TN,FP,FN
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(yTest,yPred)
#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier as kc
from sklearn.model_selection import cross_val_score as cvs
def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=kc(build_fn=build_classifier,batch_size=10,epochs=100)
accuracies=cvs(estimator=classifier,X=xTrain,y=yTrain,cv=10,n_jobs=-1)
mean=accuracies.mean()
variance=accuracies.std()
#Hyper parameter tuning
from keras.wrappers.scikit_learn import KerasClassifier as kc
from sklearn.model_selection import GridSearchCV as gs
def build_classifier():
    classifier=Sequential(optimizer)
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))
    classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=kc(build_fn=build_classifier)
params={'batch_size':[25,32],'epochs':[100,500],'optimizer':['adam','rmsprop']}
gSearch=gs(estimator=classifier,param_grid=params,scoring='accuracy',cv=10)
gSearch=gSearch.fit(xTrain,yTrain)
best_params=gSearch.best_params_
best_accuracy=gSearch.best_score_












