#!/usr/bin/env python
# coding: utf-8

# In[16]:


get_ipython().system('pip3 install xgboost')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# In[18]:


house_price = sklearn.datasets.load_boston()


# In[19]:


print(house_price)


# In[20]:


house_dataframe =pd.DataFrame(house_price.data,columns=house_price.feature_names)


# In[21]:


house_dataframe.head()


# In[22]:


house_dataframe['price']=house_price.target


# In[23]:


house_dataframe.head()


# In[24]:


house_dataframe.shape


# In[25]:


house_dataframe.isnull().sum()x


# In[26]:


house_dataframe.describe()


# In[27]:


correlation = house_dataframe.corr()


# In[28]:


plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8},cmap="Blues")


# In[29]:


x=house_dataframe.drop(['price'],axis=1)
y=house_dataframe['price']


# In[30]:


print(x)
print(y)


# In[31]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[32]:


print(x.shape,x_train.shape,x_test.shape)


# In[33]:


model=XGBRegressor()


# In[34]:


model.fit(x_train,y_train)


# In[35]:


training_data_prediction=model.predict(x_train)


# In[36]:


print(training_data_prediction)


# In[37]:


score_1=metrics.r2_score(y_train,training_data_prediction)


# In[38]:


score_2=metrics.mean_absolute_error(y_train,training_data_prediction)


# In[39]:


print("r squared error;",score_1)


# In[42]:


print("mean absolute error;",score_2)


# In[43]:


test_data_prediction=model.predict(x_test)


# In[44]:


score_1=metrics.r2_score(y_test,test_data_prediction)
score_2=metrics.mean_absolute_error(y_test,test_data_prediction)
print("r squared error;",score_1)
print("mean absolute error;",score_2)


# In[45]:


plt.scatter(y_train,training_data_prediction)
plt.xlabel("actual price")
plt.ylabel("predicted prices")
plt.title("actual price vs predicted prices")
plt.show()


# In[ ]:




