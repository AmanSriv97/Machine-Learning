%tensorflow_version 1.x

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split                        ## Imports

import keras

prot_len = 17
n_gram = 2

print("Enter the train file location")
s= input()
print("Enter the test file location")
te= input()

train = pd.read_csv()         ##Read Dataframe
test = pd.read_csv()


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




print("Hello")
