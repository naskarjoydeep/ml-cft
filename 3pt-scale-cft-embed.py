# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 01:12:36 2020

@author: Joydeep Naskar, Aditya Kumar
"""
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import sys, os
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL']= '2'


test_size=0.1
## create 3d point clouds x2, x2, x3

npoints=1000
ndelta=1000
dim=3

x1=np.random.random((npoints,dim))
x2=np.random.random((npoints,dim))
x3=np.random.random((npoints,dim))

temp1 = np.zeros((npoints,1)) + 1
temp2 = np.zeros((npoints,1)) 

#norm
def r(array):
    return (array*array).sum(axis=1)


temp2[:,0] = r(x1)
x1 = np.concatenate((temp2,x1),axis=1)
x1 = np.concatenate((temp1,x1),axis=1)

temp2[:,0] = r(x2)
x2 = np.concatenate((temp2,x2),axis=1)
x2 = np.concatenate((temp1,x2),axis=1)

temp2[:,0] = r(x3)
x3 = np.concatenate((temp2,x3),axis=1)
x3 = np.concatenate((temp1,x3),axis=1)

del(temp1,temp2)


#scaling dimensions
D1 = np.random.random((ndelta))
D2 = np.random.random((ndelta))
D3 = np.random.random((ndelta))

#dot product
def dot(array1, array2):
    return (array1*array2).sum(axis=1)

x12 = dot(x1,x2)
x23 = dot(x2,x3)
x13 = dot(x1,x3)


#cft 3-pt function in embedding space
def cft_three_point(x12,x23,x13,D1,D2,D3,coupling=1):
    D123=0.5*(D1+D2-D3)
    D231=0.5*(D2+D3-D1)
    D312=0.5*(D3+D1-D2)
    return ((-1)**dim)*(2**dim)*(x12**D123)*(x23**D231)*(x13**D312)

#scale 3-pt function in embedding space
def scale_three_point(x12,x23,x13,D1,D2,D3,coupling=1):
    a=np.random.random()
    b=np.random.random()
    c=np.random.random()
    Dtot=D1+D2+D3
    tot=a+b+c
    a=a*Dtot/tot
    b=b*Dtot/tot
    c=c*Dtot/tot
    return ((-1)**dim)*(2**dim)*(x12**a)*(x23**b)*(x13**c)

X=[]
Y=[]
num_classes=2 ## conformal invariant or not

## create the dataset
for i in range(ndelta):
    d1=D1[i]*np.ones((npoints))
    d2=D2[i]*np.ones((npoints))
    d3=D3[i]*np.ones((npoints))
    cft=cft_three_point(x12,x23,x13,d1,d2,d3)
    scale=scale_three_point(x12,x23,x13,d1,d2,d3)
    X.append([x12,x23,x13,d1,d2,d3,cft])
    Y.append([1])
    X.append([x12,x23,x13,d1,d2,d3,scale])
    Y.append([-1])
    del(d1,d2,d3,cft,scale)


## Each entry X[i] above is a point cloud, labelled by 1 or -1
## We have a sample_size number of point clouds

X=np.array(X)
Y=np.array(Y)
print(X.shape)
## reshape this data into row of features to pass to Keras for every sample
X = X.reshape(X.shape[0], 7*npoints)

X_train, X_test, Y_train, Y_test=\
         train_test_split(X,Y,test_size=test_size,train_size=1-test_size)
Y_train=keras.utils.to_categorical(Y_train)
Y_test=keras.utils.to_categorical(Y_test)
del(X,Y)

print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)


## Define the Neural Net and its Architecture
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D

layer_one=40
layer_two=10

def create_DNN():
    ## instantiate model
    model = Sequential()
    ## add a dense all-to-all relu layer
    model.add(Dense(layer_one,input_shape=(7*npoints,),activation='relu'))
    ## add a dense all-to-all relu layer
    model.add(Dense(layer_two, activation='relu'))
    ## apply dropout with rate 0.5
    model.add(Dropout(0.5))
    ## soft-max layer
    model.add(Dense(num_classes,activation='softmax'))
    return model

print('Model architecture created successfully')

## Choose the optimizer and the cost function
def compile_model(optimizer=keras.optimizers.Adam()):
    ## create the model
    model = create_DNN()
    model.compile(loss=keras.losses.categorical_crossentropy,\
                  optimizer=optimizer, metrics = ['accuracy'])
    return model

print('Model compiled successfully and ready to be trained')

## train the model in minibatches
batch_size=1000
epochs=8

##create the DNN
model_DNN = compile_model()

## train DNN and store training info in history
history=model_DNN.fit(X_train,Y_train,batch_size=batch_size,epochs=epochs,\
                      verbose=1,validation_data=(X_test,Y_test))
      
## evaluate model performance on unseen test data

# evaluate model
score = model_DNN.evaluate(X_test, Y_test, verbose=1)

# print performance
print()
print('Test loss:', score[0])
print('Test accuracy:', score[1])



## look into training history

## summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('model accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

## summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('model loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()













