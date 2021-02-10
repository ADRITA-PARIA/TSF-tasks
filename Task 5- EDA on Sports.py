#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[18]:


import numpy as np
import pandas as pd
import pandas_profiling as pp 


# In[19]:


deliveries=pd.read_csv("C:\\Users\\adrit\\Desktop\\Indian Premier League\\deliveries.csv")
deliveries.head(5)


# In[20]:


matches=pd.read_csv("C:\\Users\\adrit\\Desktop\\Indian Premier League\\matches.csv")
matches.head(5)


# In[21]:


df=pd.merge(deliveries,matches,left_on='match_id',right_on='id')
df.info()


# In[22]:


df.describe()


# In[23]:


profile=ProfileReport(df,title="Exploratory Data Analysis-Sports",explorative=True)


# In[25]:


pp.ProfileReport(df,title="Exploratory Data Analysis-Sports",explorative=True)


# # Thank you
