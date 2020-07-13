#!/usr/bin/env python
# coding: utf-8

# In[7]:


data


# In[5]:



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import pickle


data=pd.read_csv("hiring.csv")

#data preprocessing

data['experience']= data["experience"].fillna("zero")

data["test_score"]=data["test_score"].fillna(data["test_score"].mean())



# In[13]:


#Pulling data for X

X= data.iloc[:,0:3]

#Pulling data for Y

Y=data.iloc[:,-1]

#Wrting a function to convert words into integergit

def convert_to_int( word):

    word_dict={"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11}

    return word_dict[word]

X["experience"]=X["experience"].apply(lambda x:convert_to_int(x) )


# In[14]:


data


# In[15]:


X


# # - Splitting Dataset
#  * Since we have a small dataset there will be no splitting

# In[17]:


import sklearn as sk
from sklearn.linear_model import LinearRegression


# In[18]:


#Instantiate an object out of the class
regressor=LinearRegression()


# In[19]:


#Fitting the Model
regressor.fit(X,Y)


# In[24]:


#Testing the Prediction
a=[0,8,9]
regressor.predict([a])


# In[25]:


#Saving the Model to the disk
pickle.dump(regressor, open('model.pkl','wb'))  #write bytes" 


# In[34]:


# #Testing the Pickle file by openeing and predictig the same values above
# pickle.load(open('model.pkl','rb')).predict([a])


# In[33]:


#bsestway
model=pickle.load(open('model.pkl','rb'))
model.predict([a])


# In[ ]:




