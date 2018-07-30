#Convolutional Neural Network
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense
#Initializing CNN
classifier=Sequential()
#Steps 1: Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
#Step 2: Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Adding 2nd layer & Max pooling
classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
#Step 3: Flattening
classifier.add(Flatten())
#Step 4: Full Connection
classifier.add(Dense(units=128,activation='relu'))
classifier.add(Dense(units=1,activation='sigmoid'))
#Compiling CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Fitting the CNN model to images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
trainingSet=train_datagen.flow_from_directory('dataset/training_set',target_size=(64,64),batch_size=32,class_mode='binary')
testSet=test_datagen.flow_from_directory('dataset/test_set', target_size=(64,64), batch_size=32, class_mode='binary')
classifier.fit_generator(trainingSet,steps_per_epoch=8000,epochs=25,validation_data=testSet,validation_steps=2000)
#Making new predictions
import numpy as np
from keras.preprocessing import image
testImage=image.load_img('',target_size=(64,64))
testImage=image.img_to_array(testImage)
testImage=np.expand_dims(testImage,axis=0)
classifier.predict(testImage)
result=classifier.predict(testImage)
trainingSet.class_indices
if result[0][0]==1:
    prediction='Doggy'
else:
    prediction='Catty'