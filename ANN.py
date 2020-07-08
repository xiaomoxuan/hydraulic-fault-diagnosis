from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time

print(tf.__version__)

# df1 = pd.read_csv('Step_0s_02m.csv')
# df2 = pd.read_csv('Step_5s_02m.csv')
# df3 = pd.read_csv('Step_0s_01m.csv')
# df4 = pd.read_csv('Step_5s_01m.csv')
# # df5 = pd.read_csv('Ramp_0s_002.csv')
# df6 = pd.read_csv('Ramp_0s_01.csv')
# df7 = pd.read_csv('Ramp_0s_005.csv')
# df128 = pd.read_csv('Ramp_4s_02.csv')
# df9 = pd.read_csv('Ramp_4s_01.csv')
# df10 = pd.read_csv('Sinus_1Hz_02m.csv')
# df11 = pd.read_csv('Sinus_1Hz_01m.csv')
# df12 = pd.read_csv('Sinus_2Hz_02m.csv')
# df13 = pd.read_csv('Sinus_2Hz_01m.csv')

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
#
# # df5 = pd.read_csv('Ramp_0s_002.csv')

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
df128 = pd.concat(frames_ramp_4s_02)

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
df13 = pd.concat(frames_sinus_2Hz_01m)

frames = [df1,df2,df3,df4,df6,df7,df128,df9,df10,df11,df12,df13]

data_original = pd.concat(frames)

data = data_original[['pA','pB','pAd','pBd','xC','xCd']]

labels = data_original['Faults']

print(labels.shape)
print(data.shape)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = 0.2, random_state=10)

sc = StandardScaler()
train_data = sc.fit_transform(train_data)
test_data = sc.transform(test_data)

split = int(0.5*test_labels.shape[0])

validation_data = test_data[:split]
validation_labels = test_labels[:split]

test_data = test_data[split:]
test_labels = test_labels[split:]

# original network
# model = keras.models.Sequential([
#     keras.layers.Dense(100, activation=tf.nn.relu, input_shape=(6,)),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(50, activation=tf.nn.relu),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(13, activation=tf.nn.softmax)
# ])


# one hidden layer
# model = keras.models.Sequential([
#     keras.layers.Dense(100, activation=tf.nn.relu, input_shape=(6,)),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(13, activation=tf.nn.softmax)
# ])

# three hidden layers
# model = keras.models.Sequential([
#     keras.layers.Dense(100, activation=tf.nn.relu, input_shape=(6,)),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(50, activation=tf.nn.relu),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(50, activation=tf.nn.relu),
#     keras.layers.Dropout(0.5),
#     keras.layers.Dense(13, activation=tf.nn.softmax)
# ])

# four hidden layers
model = keras.models.Sequential([
    keras.layers.Dense(100, activation=tf.nn.relu, input_shape=(6,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(50, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(50, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(50, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(13, activation=tf.nn.softmax)
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              # loss='binary_crossentropy',
              metrics=['accuracy'])

print('Training---------------')

tic = time.time()

history = model.fit(train_data, train_labels,
                                  epochs=10,
                                  batch_size=128,
                                  validation_data=(validation_data, validation_labels))

toc = time.time()

print("Total Training Time is:" + str((toc-tic)/60) + " minutes")

model.save('ann_4_10_128.h5')

test_loss, test_acc = model.evaluate(test_data, test_labels)

print('Test accuracy:', test_acc)

prediction = model.predict(test_data)
y_pred = np.argmax(prediction, axis=1)

print('Confusion Matrix')
print(confusion_matrix(test_labels, y_pred))
print('Classification Report')
target_names = ['No Faults', 'Internal', 'Rod External','A External','B External','Rod Damage',
                'Increasing Friction','Particle Rod Surface','Fluid Damage Seals','Gas Enclosure',
                'Cylinder Drift','Spool Jammed 1', 'Spool Jammed 2']
print(classification_report(test_labels, y_pred, target_names=target_names, digits=4))

# Visualization of the results
history_dict = history.history
history_dict.keys()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Save acc, val_acc, loss, val_loss
np_acc = np.array(acc)
np.savetxt('ann_acc_4_10_128.txt', np_acc)
np_val_acc = np.array(val_acc)
np.savetxt('ann_val_acc_4_10_128.txt', np_val_acc)
np_loss = np.array(loss)
np.savetxt('ann_loss_4_10_128.txt', np_loss)
np_val_loss = np.array(val_loss)
np.savetxt('ann_val_loss_4_10_128.txt', np_val_loss)

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
plt.ylabel('Accuracy')
plt.legend()

plt.show()