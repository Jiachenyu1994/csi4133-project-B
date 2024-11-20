import random

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt

random.seed(42)

#data read csv
data = pd.read_csv('dataset/HandLandmarks.csv')
cols=["target"]+["handness"]+[f"landmarks {i}_{j}" for i in range(0,21)for j in ["x","y","z"] ]
data.columns=cols
data_y=data[["target"]]
data_y = tf.one_hot(data["target"].to_numpy().flatten(), 12).numpy()
data_x=data.drop(["target"],axis=1)
x_train,x_valid,y_train,y_valid=train_test_split(data_x,data_y,test_size=0.4,random_state=42)


#train model
model=tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu',input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(12, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],)
history=model.fit(x_train,y_train,validation_data=(x_valid,y_valid),epochs=200,batch_size=32)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot( history.history['loss'], label='train loss',color='blue')
plt.plot( history.history['val_loss'], label='validation loss',color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Training and Validation Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train acc',color='blue')
plt.plot(history.history['val_accuracy'], label='validation acc',color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title(f'Training and Validation Accuracy')
plt.legend()
plt.show()

model.save('models/model_rev1.h5')



