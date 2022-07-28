# -*- coding: utf-8 -*-
"""Protein Classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lq0nY7E3_TK6kNZlPs0-W1tDoYwTYVwF
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split                        ## Imports

import keras

prot_len = 17
n_gram = 2                ##Parameters

train = pd.read_csv("/content/drive/MyDrive/Protein Classification/train_data.csv")         ##Read Dataframe

test = pd.read_csv("/content/drive/MyDrive/Protein Classification/test_data.csv")

class Encoder:
  def __init__(self):
    self.A = "HRK"
    self.B = "DENQ"
    self.C = "C"
    self.D = "STPAG"
    self.E = "MILV"
    self.F = "FYW"

  def ind(self, substr, n):
    res = 0
    for i in range(len(substr)):
      res *= 6
      res += ord(substr[i])-ord('A')
    return res
  
  def group_encode(self, prot):
    encoded = ""
    for i in prot:
      if i in self.A:
        encoded += "A"
      elif i in self.B:
        encoded += "B"
      elif i in self.C:                                                                             ##Encoder Class
        encoded += "C"
      elif i in self.D:
        encoded += "D"
      elif i in self.E:
        encoded += "E"
      else:
        encoded += "F"
    
    return encoded
  
  def n_gram_encode(self, prot, n):
    group_encoded_prot = self.group_encode(prot)
    encoded = [0]*(6**n)
    for i in range(len(group_encoded_prot)-n+1):
      encoded[self.ind(group_encoded_prot[i:i+n], n)] += 1
    
    return encoded

enc = Encoder()
X = train.to_numpy()[:, 1]
X_encoded = [enc.n_gram_encode(i, n_gram) for i in X]                           ## Encodeing Dataset
Y = train.to_numpy()[:, 0]
Y = [[1, 0] if i == 0 else [0, 1] for i in Y]

for i in range(len(Y)):
  if Y[i][0] == 0:
    for j in range(16):                                                         ## Balancing Dataset
      Y.append(Y[i])
      X_encoded.append(X_encoded[i])

X_encoded = np.array(X_encoded).reshape(-1, 6**n_gram)
Y = np.array(Y).reshape(-1, 2)

X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, test_size=0.30, random_state = 101)         ## Splitting Dataset

model = keras.models.Sequential()

model.add(keras.layers.Dense(units = 72, activation='relu', input_shape=(6**n_gram,)))
model.add(keras.layers.Dense(units = 36, activation='relu'))
model.add(keras.layers.Dense(units = 18, activation='relu'))                                                ## Model Structure
model.add(keras.layers.Dense(units = 9, activation='relu'))
model.add(keras.layers.Dense(units = 2, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(1e-5), loss = 'binary_crossentropy', metrics=['binary_accuracy'])

model.summary()

model.fit(X_train, Y_train, epochs=50, validation_data=(X_test, Y_test))                ## Training Model

X_val = test.to_numpy()[:, 1]
X_val_encoded = np.array([enc.n_gram_encode(i, n_gram) for i in X_val]).reshape(-1, 6**n_gram)        ##Encoding test dataset

predictions = model.predict(X_val_encoded)

predictions = np.array([0 if i[0] >= 0.5 else 1 for i in predictions])

sample['Label'] = predictions[:,1].reshape(-1, 1)

sample.to_csv("/content/drive/MyDrive/Protein Classification/result_keras_balanced_prob_values.csv", index=False)       ## Saving the result to csv

