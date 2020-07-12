import pandas as pd
import matplotlib.pyplot as plt
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

#Wrting a function to convert words into integergit
def convert_to_int( word):
    word_dict={"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
    return word_dict[word]



X["experience"]=X["experience"].apply(lambda x: convert_to_int(x) )