#!/usr/bin/env python
# coding: utf-8

# # Data Science And Business Analytics Intern At The Sparks Foundation

# # GRIPFEB21

# # Author: Adrita Paria

# # Task 2=Prediction Using Unsupervised ML

# ### Problem statement= To predict the optimum level of clusters and represent it visually

# # Step 1: Importing standard ML libraries

# In[49]:


import pandas as pd
from pycaret.datasets import get_data
import seaborn as sns
import matplotlib.pyplot as plt


# # Step 2: Importing the data set

# In[33]:


data=pd.read_csv("C:\\Users\\adrit\\Desktop\\iris_csv.csv")
data.head(5)


# # Step 3: Visualisation Of the Dataset using Seaborn

# In[34]:


sns.pairplot(data)


# # Step 4:Setting up environment

# In[35]:


from pycaret.clustering import * 


# In[36]:


cluster= setup(data,normalize=True,session_id=123)


# # Step 5:Create model

# In[37]:


kmeans=create_model("kmeans")
print(kmeans)


# In[38]:


kmodes=create_model('kmodes')
print(kmodes)


# # Step 6: Analyze the model

# In[41]:


plot_model(kmeans,plot='elbow')


# ### From the elbow method it is clearly seen that the optimum number of clusters =4

# In[57]:


plot_model(kmeans)


# In[40]:


plot_model(kmeans,plot='tsne')


# In[42]:


plot_model(kmeans,plot='silhouette')


# In[43]:


plot_model(kmeans,plot='distribution',feature='petallength')


# In[44]:


plot_model(kmeans,plot='distribution',feature='sepallength')


# In[54]:


plot_model(kmeans,plot='distribution',feature='petalwidth')


# In[45]:


plot_model(kmeans,plot='distribution',feature='sepalwidth')


# # Final Step: Assign Cluster labels to dataset

# In[56]:


k_results=assign_model(kmeans)
k_results.head(150)
k_results.tail(100)

