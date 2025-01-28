#!/usr/bin/env python
# coding: utf-8

# In[47]:


import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


# In[6]:


Data = pd.read_csv('heart_disease_data.csv')


# In[9]:


Data.info()


# In[11]:


Data.isnull()


# In[13]:


Data.shape


# In[15]:


Data.head()


# In[17]:


Data.tail()


# In[19]:


Data.describe()


# In[22]:


Data['target'].value_counts()

1 : Defective Heart 
2 : Healthy heart
# In[25]:


x = Data.drop(columns='target',axis=1)


# In[27]:


x


# In[33]:


y = Data['target']


# In[35]:


y

Splitting train_test data
# In[38]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify = y,random_state=23)


# In[42]:


print(x_train.shape , x_test.shape)


# Model training 

# In[44]:


model = LogisticRegression()


# In[49]:


model.fit(x_train , y_train)


# In[51]:


y_pred = model.predict(x_test)


# In[55]:


y_train


# In[58]:


score = accuracy_score(y_test,y_pred)


# In[60]:


print('The accuracy score is', score)


# Building a Predictive Data

# In[62]:


input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)
#change the input_data into array

arr = np.array(input_data)


# In[64]:


print(arr)


# In[67]:


prediction = model.predict(arr.reshape(1,-1))


# In[68]:


if prediction == 0:
    print("you are healthy")
else:
    print("consult your doctor")

