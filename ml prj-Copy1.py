#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import sklearn
from sklearn.preprocessing import scale

import math
lung_data=pd.read_csv("lung_cancer_examples.csv")#import all,and the dataset file
lung_data.head(20)#show first 20 rows


# In[31]:


lung_data.corr


# In[32]:


from sklearn.model_selection import train_test_split
feature_columns=['Age','Smokes','AreaQ','Alcohol',]
predicted_class=['Result']#separating parts for testing and training


# In[33]:


x=lung_data[feature_columns].values
y=lung_data[predicted_class].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.85,random_state=1)
#allocating 85% to training and the rest to testing


# In[34]:


from sklearn.preprocessing import Imputer
fill_values =Imputer(missing_values=0,strategy='mean',axis=0)
x_train=fill_values.fit_transform(x_train)
x_test=fill_values.fit_transform(x_test)


# In[35]:


from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=1)
random_forest_model.fit(x_train,y_train.ravel())
#Applying the Algorithm


# In[36]:


#Predict
predict_train_data = random_forest_model.predict(x_test)
from sklearn import metrics
print("ACCURACY={0:.3f}".format (metrics.accuracy_score(y_test,predict_train_data)))


# In[37]:


from sklearn.linear_model import LogisticRegression
#to implement logistic regression


# logmodel=LogisticRegression()

# In[38]:


logmodel=LogisticRegression()


# In[39]:


logmodel.fit(x_train,y_train)


# In[40]:


predictions=logmodel.predict(x_test)


# In[41]:


from sklearn.metrics import classification_report


# In[42]:


classification_report(y_test,predictions)


# In[43]:


from sklearn.metrics import confusion_matrix


# In[44]:


confusion_matrix(y_test,predictions)


# In[45]:


from sklearn.metrics import accuracy_score


# In[46]:


accuracy_score(y_test,predictions)


# In[47]:


lung_data["Age"].plot.hist()


# In[48]:


lung_data.corr()


# In[49]:


plt.scatter(lung_data.Age,lung_data.Result,marker='+',color='red')


# In[51]:


sns.countplot(x="Age",data=lung_data,palette="Set3")
sns.set(rc={'figure.figsize':(14.7,8.27)})


# In[ ]:





# In[ ]:





# In[ ]:




