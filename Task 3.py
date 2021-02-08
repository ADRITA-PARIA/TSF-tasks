#!/usr/bin/env python
# coding: utf-8

# # Data Science And Business Analytics Intern At The Sparks Foundation

# # GRIPFEB21

# # Author: Adrita Paria

# # Task 3=Exploratory Data Analysis(Sample Superstore)

# # Step 1: Importing standard ML libraries

# In[132]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


#  # Step 2: Importing the data set

# In[52]:


df=pd.read_csv("C:\\Users\\adrit\\Desktop\\SampleSuperstore.csv")
df.head(5)


# # 3. Data Analysis

# In[53]:


df.describe()


# In[54]:


df.shape


# In[55]:


df.columns


# In[56]:


df.nunique()


# In[57]:


df.isnull().sum()


# # 4. Data Visualization Using Correlation Matrix

# In[58]:


correlation=df.corr()


# In[59]:


plt.figure(figsize=(10,8))
sns.heatmap(correlation,annot=True)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.show()


# In[60]:


plt.figure(figsize=(15,7))
sns.boxplot(data=df)


# # There are no outliers present here

# In[61]:


df.groupby(['Category','Sub-Category'])['Quantity'].count()


# In[62]:


sns.catplot(x='Quantity',kind='box',data=df)


# In[79]:


sns.pairplot(df)


# In[75]:


y=df[['Ship Mode','Sales']]
print(y)


# # 5. Individual visualisation of the categories

# In[93]:


plt.figure(figsize=(10,8))
sns.countplot(x=df['Segment'])
print(df['Segment'].value_counts())


# ## The section consumer shows the highest consumption rather than corporate and home office

# In[94]:


plt.figure(figsize=(10,8))
sns.countplot(x=df['Region'])
print(df['Region'].value_counts())


# ## The section west shows the highest count 

# In[90]:


plt.figure(figsize=(10,8))
sns.countplot(x=df['Category'])
print(df['Category'].value_counts())


# In[92]:


plt.figure(figsize=(15,9))
sns.countplot(x=df['Sub-Category'])
print(df['Sub-Category'].value_counts())


# In[97]:


plt.figure(figsize=(20,5))
sns.countplot(x=df['City'])
print(df['City'].value_counts())


# In[78]:


plt.figure(figsize=(10,7))
plt.bar(x=y['Ship Mode'],height=y['Sales'])
plt.title('Shipping mode vs Sales',fontsize=14)
plt.xlabel('Shipping mode',fontsize=13)
plt.ylabel('Sales',fontsize=13)
plt.show()


# In[100]:


plt.figure(figsize=(19,7))
plt.bar('Sub-Category','Category',data=df,color='y')


# In[105]:


plt.figure(figsize=(15,8))
sns.countplot(x='Sub-Category',hue='Region',data=df)
print(df['Profit'].value_counts())


# In[109]:


plt.figure(figsize=(15,8))
sns.countplot(x='Sub-Category',hue='Segment',data=df)


# In[130]:


plt.figure(figsize=(10,8))
df['Category'].value_counts().plot.pie()
plt.show()


# In[126]:


plt.figure(figsize=(10,8))
df['Sub-Category'].value_counts().plot.pie()
plt.show()


# In[139]:


fig=px.sunburst(df,path=['Country','Category','Sub-Category'],values='Sales',color='Category',hover_data=['Sales','Quantity','Profit'])
fig.update_layout(height=800)
fig.show()


# # Interpretation
# ### The dataset is about a superstore’s sales. 
# 2) The shape of the dataset is 9994, 13(Rows, Columns). 
# 3) The max profit on a single sale is 8399.976. 
# 4) Different Types of Shipping Mode is (Standard, Second, First Class and Same Day).
# 5) Categories of the Products are (Office Supplies, Technology, and Furniture).
# 6) Segment of the Customers are (Consumer, Home Office, Corporate).
# 7) Products are delivered in 39 states. 
# 8) Region of services is (East, West, South, and Central).
# 9) There are several Sub-Categories of Products.
# 10) Total profit made: 286397.0217. 
#     11) Total Sales made: 2297200.8603. 
#         12) Therefore Profit Percentage is: 12.46.
#             13) Many Discounts are also given. 
#             14) As California has the highest number of sales let’s take a look its stats. 
#             15) In California Office Supplies has the highest number of sales. 
#             16) In Office Supplies Paper is the highest Sold. 
#             17) Most people in California prefer Standard Class of shipping mode.
#             18) As most of the sales are in the quantity of 3, company should provide a special discount on a bundle of 3, so sales may increase. 
#             19) In west also Office supplies has the highest sales. 
#             20) But a noticeable thing to see is that instead of paper in west highest sales is of binders.
#             21) In the technology category Phones has the highest sales in California. 
#             22) Sales in Furniture : 74199.7953 
#                 23) Sales in Office Supplies: 719047.032 
#                     24) Sales in Technology: 836154.033 
#                         25) Though highest no.of sales were of Office Supplies still Technology’s Sales are greater in number. 
#                         26) Profit in Furniture: 18451.272 
#                             27) Profit in Office Supplies: 122490.8008 
#                                 28) Profit in Technology: 145454.9481 
#                                     29) There is highest loss in the Office Supplies Category. 
#                                     30) California is the state where there is highest profit in the art category so art related ads should be run in California. 
#                                     31) Company should work on the Central region because that region has the highest losses. 
#                                     32) And state wise company should work on Texas.
