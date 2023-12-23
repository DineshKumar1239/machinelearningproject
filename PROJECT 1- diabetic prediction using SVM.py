#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[4]:


diabetes=pd.read_csv('diabetes.csv')


# In[5]:


get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# In[6]:


diabetes.head()


# In[8]:


diabetes.shape


# In[11]:


diabetes.describe()


# In[16]:


diabetes['Outcome'].value_counts()


# In[19]:


diabetes.groupby('Outcome').mean()


# In[24]:


x=diabetes.drop(columns='Outcome',axis=1)
y=diabetes['Outcome']


# In[25]:


print(x)


# In[26]:


print(y)


# In[32]:


scale=StandardScaler()


# In[ ]:





# In[34]:


scale.fit(x)


# In[35]:


scaled=scale.transform(x)


# In[36]:


print(scaled)


# In[37]:


x =scaled


# In[38]:


print(x,y)


# In[40]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[42]:


print(x.shape,x_train.shape,x_test.shape)


# In[43]:


classifier = svm.SVC(kernel='linear')


# In[45]:


classifier.fit(x_train,y_train)


# In[47]:


x_test_predict=classifier.predict(x_test)
test_data_accuracy=accuracy_score(x_test_predict,y_test)


# In[48]:


print('test accuracy is;',test_data_accuracy)


# In[49]:


x_train_predict=classifier.predict(x_train)
train_data_accuracy=accuracy_score(x_train_predict,y_train)


# In[50]:


print('train accuracy is;',train_data_accuracy)


# In[51]:


# making predictive system


# In[53]:


input_data=(10,168,74,0,0,38,0.537,34)
input_as_array =np.asarray(input_data)
input_reshape =input_as_array.reshape(1,-1)
std_data = scale.transform(input_reshape)
print(std_data)

predict=classifier.predict(std_data)
print(predict)

if (predict[0]==0):
    print("the person is not diabetic ")
else:
    print("the person is diabetic")


# In[ ]:





# In[ ]:





# In[ ]:




