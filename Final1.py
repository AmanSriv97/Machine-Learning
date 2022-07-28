#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries
from tensorflow import keras
import sklearn
import mysql.connector
from mysql.connector import Error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


# In[2]:


import tensorflow as tf
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score


# # Database Connection

# In[3]:


class DB_Conn:
    def __init__(self):
        self.connection = None
        self.__host_name = "localhost"
        self.__username = "root"
        self.__pw="Root@123"  # SQL Terminal Password ####  ### Class Attributes
        self.db="ACP_detection"  # Databse Name #####
    
    def set_credentials(self, username, password, database):
        self.username = username
        self.pw = password
        self.db = database
        
    def create_server_connection(self):
        try:
            connection= mysql.connector.connect(
                host=self.__host_name,
                user=self.__username,
                password=self.__pw
            )
            print("Server connection successful")
        except Error as err:
            print(f"Error: '{err}'")
        return connection
    
    def create_db_connection(self):
        connection= None
        try:
            connection= mysql.connector.connect(
                host=self.__host_name,
                user=self.__username,
                password=self.__pw,
                database= self.db)
            print("Database connection successful")
        except Error as err:
            print(f"Error: '{err}'" )
        return connection


# # Database Creation using Query

# In[4]:


class DB_Manager:
    def __init__(self):
        self.db_create_query= "Create database ACP_detection" ### Database creation Query
    
    def get_db_name(self, db_name):
        self.db_create_query = "Create database " + db_name
    
    def create_database(self, connection):
        cursor= connection.cursor()
        try:
            cursor.execute(self.db_create_query)
            print("Database created successfully")
        except Error as err:
            print(f"Error: '{err}'" )
    
    def execute_query(self,connection,query):
        cursor= connection.cursor()
        try:
            cursor.execute(query)
            result = cursor.fetchall()
            connection.commit()
            print("Query was successful")
        except Error as err:
            print(f"Error: '{err}'" )
        return result
    
    def load_data(self, connection, data):
        cur= connection.cursor()
        s= "INSERT INTO samples (Peptide_sequence, label) VALUES (%s,%s)"
        cur.executemany(s, data)
        connection.commit()
        print("Data Loaded Successfully")


# # Accessing Database using Objects

# In[5]:


DBC = DB_Conn()


# In[6]:


server_connection = DBC.create_server_connection()


# In[7]:


DBM = DB_Manager()
DBM.create_database(server_connection)


# In[8]:


db_connection = DBC.create_db_connection()


# In[9]:


create_table= """
create table samples(
Peptide_sequence varchar(200),
label int);
"""
#connect to the database
DBM.execute_query(db_connection,create_table)


# # Reading Fasta files

# In[10]:


# Reading the input Fasta File

#Negative samples
with open(r'C:\Users\mailt\Desktop\fasta\fasta\data\negative\balanced.fasta','r') as a:
    x = a.read()
    
#Positive Samples
with open(r'C:\Users\mailt\Desktop\fasta\fasta\data\positive\balanced.fasta','r') as e:
    u= e.read()


# # Data Pre processing 

# In[11]:


# Data Pre processing 

class Data_Preprocessor:
    def __init__(self):
        pass
    
    def preprocess(self, data, cls): 
        s= data.split("\n")
        n= len(s)

        arr=[]
        while n>0:
            arr.append(s[n-2])
            n= n-2
        
        lst=[]
        for i in range(len(arr)-1):
            lst1=[]
            a=()   
            lst1.append(arr[i])
            lst1.append(cls)
            a= tuple(lst1)
            lst.append(a)

        return lst


# In[12]:


preprocessor = Data_Preprocessor()
lst = preprocessor.preprocess(x, 0)
lst1 = preprocessor.preprocess(u, 1)



# # Loading Samples in SQL Table

# In[13]:


DBM.load_data(db_connection, lst)         ## Negative Samples
DBM.load_data(db_connection, lst1)        ##Positive Samples


# In[14]:


q1="""
select * from samples;
"""
results=DBM.execute_query(db_connection, q1)


# In[15]:


# Converting the data to a dataframe.

#create data frame 

df=[]

for res in results:
    result= list(res)
    df.append(res)
    
columns= ["Peptide_sequence", "label"]
df= pd.DataFrame(df,columns=columns )

display(df)


# In[16]:


df = df.reindex(np.random.permutation(df.index)) #### Shuffling the Dataframe
n_gram=2


# In[17]:


# Loading the p-feature analysis file

d2=pd.read_csv(r"C:\Users\mailt\Desktop\OOPD project\only_dipeptid (1).csv")
d2.head()
pf = d2.reindex(np.random.permutation(d2.index))


# In[18]:


#Preprocessing
sc=StandardScaler()
c2=pd.DataFrame(sc.fit_transform(pf))
c2.columns=pf.columns
c2.head()


# In[19]:


Y1=c2.Label


# # Feature Selection using Variance Threshold

# In[20]:


vt=VarianceThreshold(1.0)
fs=vt.fit_transform(c2)
fs=c2.columns[vt.get_support(indices=True)]
c2[fs].head()
c3=c2[fs]
c3.head()


# In[21]:


m=round(c3.mean(),6)
m
c3.fillna(m,inplace=True)
c3.head(5)


# In[22]:


from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(c3, Y1, test_size=0.30, random_state= 22) 


# # Decision Tree Classifier

# In[23]:


#Decision taken after computing all features 
cl1=DecisionTreeClassifier(random_state=15)
cl1=cl1.fit(X_train,Y_train)
cl1


# In[24]:


prediction_dt=cl1.predict(X_test)


# In[25]:


as1=(accuracy_score(prediction_dt,Y_test))
print("Accuracy of Decision Tree: ",as1)
fs1=(f1_score(prediction_dt,Y_test))
print("f1_score of Decision Tree: ",fs1)


# # Random Forest Classifier

# In[26]:


model_Random = RandomForestClassifier(random_state=18) #max_depth=2,max_features='auto',n_estimators=3, 

model_Random.fit(X_train,Y_train)


# In[27]:


model_Random.score(X_train,Y_train)


# In[28]:


prediction_RF=model_Random.predict(X_test)


# In[29]:


Y_test_pred_proba = model_Random.predict_proba(X_test)[:,1]
Y_test_pred_proba


# In[30]:


as2=(accuracy_score(prediction_RF,Y_test))

fs2=(f1_score(prediction_RF,Y_test))
print("f1_score of RF: ",fs2)
print("Accuracy of RF: ",as2)


# In[31]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

print('confusion matrix')
print(confusion_matrix(Y_test, prediction_RF))
    #confusion matrix
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(confusion_matrix(Y_test, prediction_RF), annot=True, fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()


# In[32]:


from sklearn.metrics import roc_curve
y_pred_keras = Y_test_pred_proba.ravel()
y_test_kears= Y_test.ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test_kears, y_pred_keras)


# In[33]:


from sklearn.metrics import auc
auc_keras = auc(fpr_keras, tpr_keras)
print(auc_keras)


# In[34]:


plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


# In[35]:


class Conf_matrix:
    # Creating a function to report confusion metrics
    def confusion_metrics (self,conf_matrix):
    # save confusion matrix and slice into four pieces
        TP = conf_matrix[1][1]
        TN = conf_matrix[0][0]
        FP = conf_matrix[0][1]
        FN = conf_matrix[1][0]
        print('True Positives:', TP)
        print('True Negatives:', TN)
        print('False Positives:', FP)
        print('False Negatives:', FN)
    
    # calculate accuracy
        conf_accuracy = (float (TP+TN) / float(TP + TN + FP + FN))
    
    # calculate mis-classification
        conf_misclassification = 1- conf_accuracy
    
    # calculate the sensitivity
        conf_sensitivity = (TP / float(TP + FN))
    # calculate the specificity
        conf_specificity = (TN / float(TN + FP))
    
        # calculate precision
        conf_precision = (TN / float(TN + FP))
        # calculate f_1 score
        conf_f1 = 2 * ((conf_precision * conf_sensitivity) / (conf_precision + conf_sensitivity))
        print('-'*50)
        print(f'Accuracy: {round(conf_accuracy,2)}') 
        print(f'Mis-Classification: {round(conf_misclassification,2)}') 
        print(f'Sensitivity: {round(conf_sensitivity,2)}') 
        print(f'Specificity: {round(conf_specificity,2)}') 
        print(f'Precision: {round(conf_precision,2)}')
        print(f'f_1 Score: {round(conf_f1,2)}')


# In[36]:


mat= Conf_matrix()
res=confusion_matrix(Y_test, prediction_RF)
mat.confusion_metrics(res)


# # Saving the values to the database

# In[37]:


create_table= """
create table Results(
Peptide_sequence varchar(200),
label int);
"""
#connect to the database
DBM.execute_query(db_connection,create_table)


# In[38]:


preprocessor = Data_Preprocessor()
aY = [1 if i == 1 else 0 for i in prediction_RF]
testindex=X_test.index

lst3=[]
for i in range(len(aY)):
    #lst2 = preprocessor.preprocess(df.Peptide_sequence[testindex], i)
    lst2=[]
    a=()
    b= testindex[i]
    lst2.append(df.Peptide_sequence[b])
    lst2.append(aY[i])
    a= tuple(lst2)
    lst3.append(a)


# ### 

# In[40]:


class DB_Load:
    def __init__(self):
        pass
    
    def load_data1(self, connection, data):
        cur= connection.cursor()
        s= "INSERT INTO results (Peptide_sequence, label) VALUES (%s,%s)"
        cur.executemany(s, data)
        connection.commit()
        print("Data Loaded Successfully")


# In[41]:


load= DB_Load()
load.load_data1(db_connection,lst3)


# In[ ]:




