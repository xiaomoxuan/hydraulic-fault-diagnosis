from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model, Sequential
from keras.utils import np_utils
from keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from pandas.core.frame import DataFrame

print(tf.__version__)

def labels_reshape(n):
    s = []
    for i in range(0,n.shape[0],6):
        s.append(n[i])
    return s

step_0s_02m = pd.read_csv('Step_0s_02m.csv')
step_0s_02m_continue = pd.read_csv('Step_0s_02m_continue.csv')
frames_step_0s_02m = [step_0s_02m,step_0s_02m_continue]
df1 = pd.concat(frames_step_0s_02m)

step_5s_02m = pd.read_csv('Step_5s_02m.csv')
step_5s_02m_continue = pd.read_csv('Step_5s_02m_continue.csv')
frames_step_5s_02m = [step_5s_02m,step_5s_02m_continue]
df2 = pd.concat(frames_step_5s_02m)

step_0s_01m = pd.read_csv('Step_0s_01m.csv')
step_0s_01m_continue = pd.read_csv('Step_0s_01m_continue.csv')
frames_step_0s_01m = [step_0s_01m,step_0s_01m_continue]
df3 = pd.concat(frames_step_0s_01m)

step_5s_01m = pd.read_csv('Step_5s_01m.csv')
step_5s_01m_continue = pd.read_csv('Step_5s_01m_continue.csv')
frames_step_5s_01m = [step_5s_01m,step_5s_01m_continue]
df4 = pd.concat(frames_step_5s_01m)

# df5 = pd.read_csv('Ramp_0s_002.csv')

ramp_0s_01 = pd.read_csv('Ramp_0s_01.csv')
ramp_0s_01_continue = pd.read_csv('Ramp_0s_01_continue.csv')
frames_ramp_0s_01 = [ramp_0s_01,ramp_0s_01_continue]
df6 = pd.concat(frames_ramp_0s_01)

ramp_0s_005 = pd.read_csv('Ramp_0s_005.csv')
ramp_0s_005_continue = pd.read_csv('Ramp_0s_005_continue.csv')
frames_ramp_0s_005 = [ramp_0s_005,ramp_0s_005_continue]
df7 = pd.concat(frames_ramp_0s_005)

ramp_4s_02 = pd.read_csv('Ramp_4s_02.csv')
ramp_4s_02_continue = pd.read_csv('Ramp_4s_02_continue.csv')
frames_ramp_4s_02 = [ramp_4s_02,ramp_4s_02_continue]
df8 = pd.concat(frames_ramp_4s_02)

ramp_4s_01 = pd.read_csv('Ramp_4s_01.csv')
ramp_4s_01_continue = pd.read_csv('Ramp_4s_01_continue.csv')
frames_ramp_4s_01 = [ramp_4s_01,ramp_4s_01_continue]
df9 = pd.concat(frames_ramp_4s_01)

sinus_1Hz_02m = pd.read_csv('Sinus_1Hz_02m.csv')
sinus_1Hz_02m_continue = pd.read_csv('Sinus_1Hz_02m_continue.csv')
frames_sinus_1Hz_02m = [sinus_1Hz_02m,sinus_1Hz_02m_continue]
df10 = pd.concat(frames_sinus_1Hz_02m)

sinus_1Hz_01m = pd.read_csv('Sinus_1Hz_01m.csv')
sinus_1Hz_01m_continue = pd.read_csv('Sinus_1Hz_01m_continue.csv')
frames_sinus_1Hz_01m = [sinus_1Hz_01m,sinus_1Hz_01m_continue]
df11 = pd.concat(frames_sinus_1Hz_01m)

sinus_2Hz_02m = pd.read_csv('Sinus_2Hz_02m.csv')
sinus_2Hz_02m_continue = pd.read_csv('Sinus_2Hz_02m_continue.csv')
frames_sinus_2Hz_02m = [sinus_2Hz_02m,sinus_2Hz_02m_continue]
df12 = pd.concat(frames_sinus_2Hz_02m)

sinus_2Hz_01m = pd.read_csv('Sinus_2Hz_01m.csv')
sinus_2Hz_01m_continue = pd.read_csv('Sinus_2Hz_01m_continue.csv')
frames_sinus_2Hz_01m = [sinus_2Hz_01m,sinus_2Hz_01m_continue]
df13 = pd.concat(rames_sinus_2Hz_01mf)

frames = [df1,df2,df3,df4,df6,df7,df8,df9,df10,df11,df12,df13]

data_original = pd.concat(frames)

data = data_original[['pA','pB','pAd','pBd','xC','xCd']]

labels = data_original['Faults']

labels = labels[::6]
print(labels.shape)

sc = StandardScaler()
data = sc.fit_transform(data)

data = data.reshape(-1,6,6,1)

print(data.shape)
print(labels.shape)

train_data, test_data, train_labels, test_labels1 = train_test_split(data, labels, test_size = 0.2, random_state=10)

split = int(0.5*test_labels1.shape[0])

validation_data = test_data[:split]
validation_labels = test_labels1[:split]

test_data = test_data[split:]
test_labels1 = test_labels1[split:]

print(train_data.shape)
print(validation_data.shape)
print(test_data.shape)

train_labels = np_utils.to_categorical(train_labels,num_classes=13)
validation_labels = np_utils.to_categorical(validation_labels,num_classes=13)
test_labels = np_utils.to_categorical(test_labels1, num_classes=13)

print(train_labels.shape)
print(validation_labels.shape)
print(test_labels.shape)

# Original CNN
model = Sequential()
model.add(Convolution2D(input_shape=(6,6,1),filters=10, kernel_size=5,
                        strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=2, strides=2, padding='same',data_format='channels_first'))
model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2,2,'same',data_format='channels_first'))
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(13))
model.add(Activation('softmax'))
adam = Adam(lr=1e-4)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# One less Conv2D, and one less MaxPooling2D
# model = Sequential()
# model.add(Convolution2D(input_shape=(6,6,1),filters=10, kernel_size=5,
#                         strides=1, padding='same', data_format='channels_first'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=2, strides=2, padding='same',data_format='channels_first'))
# model.add(Flatten())
# model.add(Dense(1024))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))
# model.add(Dense(13))
# model.add(Activation('softmax'))
# adam = Adam(lr=1e-4)
# model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
# model.summary()

# One more Conv2D, and one more MaxPooling2D
# model = Sequential()
# model.add(Convolution2D(input_shape=(6,6,1),filters=10, kernel_size=5,
#                         strides=1, padding='same', data_format='channels_first'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=2, strides=2, padding='same',data_format='channels_first'))
# model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2,'same',data_format='channels_first'))
#
# model.add(Convolution2D(32, 5, strides=1, padding='same', data_format='channels_first'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2,'same',data_format='channels_first'))
#
# model.add(Flatten())
# model.add(Dense(1024))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))
# model.add(Dense(13))
# model.add(Activation('softmax'))
# adam = Adam(lr=1e-4)
# model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
# model.summary()

# Two more Conv2D, and Two more MaxPooling2D
# model = Sequential()
# model.add(Convolution2D(input_shape=(6,6,1),filters=10, kernel_size=5,
#                         strides=1, padding='same', data_format='channels_first'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=2, strides=2, padding='same',data_format='channels_first'))
# model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2,'same',data_format='channels_first'))
#
# model.add(Convolution2D(32, 5, strides=1, padding='same', data_format='channels_first'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2,'same',data_format='channels_first'))
#
# model.add(Convolution2D(16, 5, strides=1, padding='same', data_format='channels_first'))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(2,2,'same',data_format='channels_first'))
#
# model.add(Flatten())
# model.add(Dense(1024))
# model.add(Activation('relu'))
# model.add(Dropout(0.1))
# model.add(Dense(13))
# model.add(Activation('softmax'))
# adam = Adam(lr=1e-4)
# model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
# model.summary()


print('Training---------------')

tic = time.time()

history = model.fit(train_data,train_labels,epochs=10, batch_size=128,validation_data=(validation_data, validation_labels))

toc = time.time()

model.save('cnn_4_10_128.h5')

print("Total Training Time is:" + str((toc-tic)/60) + " minutes")

print('Testing---------------')
loss, accuracy = model.evaluate(test_data,test_labels)
print('test loss: ', loss)
print('test accuracy: ', accuracy)

prediction = model.predict(test_data)
y_pred = np.argmax(prediction, axis=1)

print(y_pred.shape)
print(test_labels1.shape)

print('Confusion Matrix')
print(confusion_matrix(test_labels1, y_pred))
print('Classification Report')
target_names = ['No Faults', 'Internal', 'Rod External','A External','B External','Rod Damage',
                'Fluid Increasing Friction','Fluid Rod Surface','Fluid Damage Seals','Gas Enclosure',
                'Cylinder Drift','Spool Jammed 1', 'Spool Jammed 2']
print(classification_report(test_labels1, y_pred, target_names=target_names, digits=4))

# Visualization of the results
history_dict = history.history
history_dict.keys()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Save acc, val_acc, loss, val_loss
np_acc = np.array(acc)
np.savetxt('cnn_acc_4_10_128.txt', np_acc)
np_val_acc = np.array(val_acc)
np.savetxt('cnn_val_acc_4_10_128.txt', np_val_acc)
np_loss = np.array(loss)
np.savetxt('cnn_loss_4_10_128.txt', np_loss)
np_val_loss = np.array(val_loss)
np.savetxt('cnn_val_loss_4_10_128.txt', np_val_loss)

epochs = range(1, len(acc) + 1)

plt.figure(1)
# "r" is for "solid red line"
plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
plt.plot(epochs, acc, 'r', label='Training acc')
# b is for "solid blue line"
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
