import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("hiring.csv")
import numpy as np
import pickle



data=pd.read_csv("hiring.csv")

#data preprocessing

data['experience']= data["experience"].fillna(0)

data["test_score"]=data["test_score"].fillna(data["test_score"].mean())



#Pulling data for X

X= data.iloc[:,0:3]

#Pulling data for Y

Y=data.iloc[:,-1]