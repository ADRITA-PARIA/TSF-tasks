#!/usr/bin/env python
# coding: utf-8

# # Data Science And Business Analytics Intern At The Sparks Foundation

# # GRIPFEB21

# # Author: Adrita Paria

# # Task 1= Prediction Using Supervised ML

# ### Problem Statement=What will be the predicted score of a student if she/he studies for 9.25 hours/day.

# # First Step :- Importing the standard ML libraries

# In[3]:


import numpy as np
import pandas as pd
from sklearn import linear_model
import statsmodels.api as sm
import matplotlib.pyplot as plt


# # Second Step : Reading Data set given by the Sparks Foundation 

# In[5]:


data=pd.read_csv("http://bit.ly/w-data")
data.head(5)


# # Step Three: Visualization of data using Scatter Plot 

# In[6]:


plt.scatter(x=data["Hours"],y=data["Scores"])
plt.title("study hours vs results",fontsize=14)
plt.ylabel("scores",fontsize=14)
plt.xlabel("hours",fontsize=14)
plt.grid(True)
plt.show()


# # Step Four: Splitting Data Variables into target and feature. Here my Feauture Variable will be Number of study hours and Target variables is the Score of students. 

# In[23]:


x=data.iloc[:, :-1].values
y=data.iloc[:,1].values
print("x values:",x)
print("y values:",y)


# # Step Five: Split data set into train and test sets 

# In[25]:


from sklearn.model_selection import train_test_split


# In[27]:


x_train,x_test,y_train,y_test= train_test_split(x, y, test_size=0.3, random_state=0)
print(x_train.shape)
print(y_train.shape)


# # Step Six: Training Simple Linear Regression Model

# In[28]:


regr=linear_model.LinearRegression()
regr.fit(x_train,y_train)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)


# # Step Seven: Visualisation of the line of fit 

# In[30]:


plt.scatter(x,y)
plt.plot(x,regr.predict(x),color='green')
plt.show()


# # Step Eight : Predicting the score for 9.24 hours/day

# In[31]:


new_x=np.array([[9.25]])
print("Prediction of the score for 9.25 hours/day:",regr.predict(new_x))


# # Step Nine= Predicting and Testing

# In[36]:


print(x_test)
ypred=regr.predict(x_test)
print("predicted:",ypred)
print("actual:",y_test)


# # Final Step : Evaluating the model

# In[37]:


x=sm.add_constant(x)
model=sm.OLS(y, x).fit()
predictions=model.predict(x)
print(model.summary())

