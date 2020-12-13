#!/usr/bin/env python
# coding: utf-8

# 
# # THE SPARK FOUNDATION - GRIP- DATA SCIENCE AND BUSINESS ANALYTICS - DEC 2020

# ### TASK 1-PREDICTION USING SUPERVISED MACHINE LEARNING

# ### AUTHOR - SRUTHI B

# #### Dataset used : Student scores

# * It can be downloaded through the following link : http://bit.ly/w-data

# ### Problem Objectives :
# * Predict the percentage of an student based on the number of study hours
# * What will be the predicted score if a student studies for 9.25 hrs per day?

# ### Import Necessary libraries

# In[3]:


import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### read the data

# In[4]:


data=pd.read_csv("C:/Users/ELCOT/Documents/data1.csv")
data.head()


# In[ ]:


data.shape  # to find dimentions


# In[7]:


data.info()


# In[8]:


data.describe()


# ### visualize the data

# In[12]:


sns.scatterplot(x=data['Hours'],y=data['scores']); #plot the data


# In[13]:


sns.regplot(x=data['Hours'],y=data['scores']);   #reg plot for better understanding


# ### Seperate data and target

# In[6]:


x=data.drop(["scores"],axis=1)
y=data.scores


# ### Train-Test split

# In[8]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# ### model building

# In[14]:


from sklearn.linear_model import LinearRegression

regressor=LinearRegression()


# In[16]:


regressor.fit(x_train,y_train)
print(regressor.coef_)
print(regressor.intercept_)


# In[18]:


line=regressor.coef_* x+regressor.intercept_
plt.scatter(x,y)
plt.plot(x,line);
plt.show()


#  ### making predictions

# In[20]:


pred_test=regressor.predict(x_test)
pred_test


# ### comparison between actual and predicted value

# In[22]:


df=pd.DataFrame({'Actual':y_test,'predicted':pred_test})
print(df)


# ### what will be the predicted score if a student study for 9.25 hrs per day?

# In[33]:


y_pred=regressor.predict([[9.25]])
print('predicted scores:',y_pred)


# ### evaluating model

# In[35]:


from sklearn import metrics
print('mean absolute error;',metrics.mean_absolute_error(y_test,pred_test))


# ## TASK 1 COMPLETED
