#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as  np
dataset=pd.read_csv("supplement1.csv")
dataset


# In[3]:


dataset.isnull().sum()
dataset


# In[4]:


dataset.describe()


# In[5]:


pip install plotly


# In[6]:


import plotly.express as px
pie=dataset["Store_Type"].value_counts()
store=pie.index
orders=pie.values
fig=px.pie(dataset,values=orders,names=store)
fig.show()


# In[7]:


pie1=dataset["Location_Type"].value_counts()
store=pie1.index
orders=pie1.values
fig=px.pie(dataset,values=orders,names=store)
fig.show()


# In[8]:


pie2=dataset["Discount"].value_counts()
discount=pie2.index
orders=pie2.values
fig=px.pie(dataset,values=orders,names=discount)
fig.show()


# In[9]:


pie3=dataset["Holiday"].value_counts()
holiday=pie3.index
orders=pie3.values
fig=px.pie(dataset,values=orders,names=holiday)
fig.show()


# In[10]:


dataset["Discount"]=dataset["Discount"].map({"No":0,"Yes":1})
dataset


# In[11]:


dataset["Store_Type"]=dataset["Store_Type"].map({"S1": 1,"S2": 2,"S3": 3,"S4": 4})
dataset


# In[12]:


dataset["Location_Type"]=dataset["Location_Type"].map({"L1": 1,"L2": 2,"L3": 3,"L4": 4,"L5": 5})
dataset


# In[13]:


dataset["Region_Code"]=dataset["Region_Code"].map({"R1": 1,"R2": 2,"R3": 3,"R4": 4})
dataset


# In[14]:


x=np.array(dataset[["Store_Type","Location_Type","Region_Code","Holiday","Discount"]])
y=np.array(dataset["#Order"])
x


# In[15]:


y


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[17]:


pip install lightgbm


# In[18]:


import lightgbm as lgb
model=lgb.LGBMRegressor()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred


# In[19]:


y_test


# In[20]:


accuracy=model.score(x_test,y_test)
print(accuracy)


# In[21]:


data=pd.DataFrame({"predicted Order":y_pred.flatten()})
data

