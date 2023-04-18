#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()



# In[10]:


data=pd.read_csv(r"D:\f1\Admission_Predict.csv")
print(data)


# In[11]:


df=data.copy()
df.tail(20)

df.drop('serial No.',axis=1,inplace=True)
df.head()
# In[12]:


df.isnull().sum()


# In[13]:


df.dtypes


# In[14]:


df.shape


# In[15]:


df.describe


# In[16]:


df.columns=df.columns.str.strip()
df.columns


# In[21]:


plt.figure(figsize=(10,6))
sns.regplot(x='GRE Score',y='Chance of Admit', data=df)


# In[23]:


from scipy import stats
p_coeff,p_value=stats.pearsonr(df['GRE Score'],df['Chance of Admit'])
print('Person Coefficient:',p_coeff)
print('p  value:',p_value)


# In[25]:


plt.figure(figsize=(10,6))
sns.regplot(x=df['University Rating'],y=df['Chance of Admit'])


# In[28]:


plt.figure(figsize=(10,6))
sns.boxplot(data=df,x='University Rating',y='Chance of Admit') 


# In[ ]:




