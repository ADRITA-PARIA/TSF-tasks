#!/usr/bin/env python
# coding: utf-8

# # Data Science And Business Analytics Intern At The Sparks Foundation

# # GRIPFEB21

# # Author: Adrita Paria

# # Task 4=Exploratory Data Analysis(Terrorism)

# # Step 1: Importing standard ML libraries

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Step 2: Importing the data set

# In[24]:


df=pd.read_csv("C:\\Users\\adrit\\Desktop\\Global Terrorism - START data\\globalterrorismdb_0718dist.csv",encoding='latin1',low_memory=False)
df.head(5)


# In[25]:


df.tail()


# # 3. Data Analysis

# In[26]:


df.describe()


# In[27]:


df.shape


# In[28]:


df.isnull().sum()


# In[29]:


df.isnull().sum().sum()


# In[30]:


df.columns.values


# In[31]:


df=df[['gname' ,'weaptype1_txt','targtype1','targtype1_txt','iyear', 'imonth', 'iday','country', 'country_txt', 'region','region_txt','provstate', 'city', 'attacktype1', 'attacktype1_txt','nkill','nwound', 'weaptype1']]
df.head()      


# # 4. Data Visualization Using Correlation Matrix

# In[32]:


plt.figure(figsize=(15,8))
correlation=df.corr()
sns.heatmap(correlation,annot=True)


# # 5. Individual visualisation of the categories

# In[12]:


plt.figure(figsize=(30,15))
sns.countplot(x=df["region_txt"])
print(df["region_txt"].value_counts())


# In[13]:


plt.figure(figsize=(30,8))
sns.countplot(x=df["country_txt"])
print(df["country_txt"].value_counts())


# In[14]:


df["country_txt"].value_counts().head(10).plot.bar(color='red')


# # Iraq is the hot zone of terrorism

# In[15]:


plt.figure(figsize=(20,8))
sns.countplot(x=df["attacktype1_txt"])
print(df["attacktype1_txt"].value_counts())


# # Explosives are mostly used weapons,follwed by firearms,unknown etc.

# In[16]:


plt.figure(figsize=(30,8))
sns.countplot(x=df["targtype1_txt"])
print(df["targtype1_txt"].value_counts())


# # Most targetted places are Private Citizens & Property,Military,police,government.

# In[17]:


df["country_txt"].value_counts().head(10).plot.pie()


# In[18]:


df["targtype1_txt"].value_counts().head(10).plot.pie()


# In[19]:


df["attacktype1_txt"].value_counts().head(10).plot.pie()


# In[20]:


plt.figure(figsize=(10,8))
df["weaptype1_txt"].value_counts().head(10).plot.pie()


# In[40]:


df["iyear"].value_counts().head(10).plot.bar(color='y')
print(df['iyear'].value_counts()[1:11])


# # Year 2014 accounts for the highest target

# In[39]:


plt.figure(figsize=(30,15))
sns.barplot(x=df["gname"].value_counts()[1:11],y=df["gname"].value_counts()[1:11].index,palette='magma')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


# # The frequent terror groups named Taliban,Islamic State of Iraq and the Levant (ISIL),etc

# # Interpretation
# 

# - **Iraq is the hot zone of terrorism**
# - **Maximum terrorism has occured in Middle East and North Africa.** 
# - **Most of the events has occured Iraq following Pakistan,Afganistan,India.**
# - **Most targetted places are Private Citizens & Property,Military,police,government.**
# - **Year 2014 accounts for the highest target.**
# - **Explosives are mostly used weapons,follwed by firearms,unknown etc.**
# - **The frequent terror groups named Taliban,Islamic State of Iraq and the Levant (ISIL),etc.**

# # Thank you
