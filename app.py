#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
from flask import Flask, request,jsonify, render_template
import pickle


# In[21]:


app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))


# In[22]:


#Expose your model ovr an api
@app.route('/')
def home():
    return render_template('index.html')


# In[23]:


@app.route('/predict',methods=['POST'])
def predict():
    """"
    For rendering results on html GUI
    """
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    output=round(prediction[0],2)
    
    return render_template('index.html', prediction_text="Employee Salary is $ {}".format(output))


# In[25]:


@app.route('/predict_api',methods=['POST'])
def predict_api():
    """
    For Direct API through request"
    """
    data=request.get_json(force=True)
    prediction=model.predict([np.array(list(data.values()))])
    output=prediction[0]


# In[26]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




