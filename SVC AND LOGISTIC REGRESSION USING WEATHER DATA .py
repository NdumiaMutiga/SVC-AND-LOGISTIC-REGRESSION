#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing packages 
import numpy as np
import pandas as pd


# In[3]:


#imporing the data 

dataset='https://raw.githubusercontent.com/TrainingByPackt/Data-Science-with-Python/master/Chapter03/weather.csv'


# In[4]:


df=pd.read_csv(dataset)
df.head()


# In[5]:


#Dummy code the categorical features 
df_dummies=pd.get_dummies(df, drop_first=True)


# In[7]:


#shuffle the dataset to avoid any ordering 
from sklearn.utils import shuffle
df_shuffled=shuffle(df_dummies, random_state=42)
dv="Rain"
X=df_shuffled.drop(dv, axis=1)
y=df_shuffled[dv]


# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33,random_state=42 )


# In[12]:


#scale the data 
from sklearn.preprocessing import StandardScaler
model=StandardScaler()
X_train_scaled=model.fit_transform(X_train)
X_test_scaled=model.transform(X_test)


# In[13]:


#using Grid Search to find the best combination of Hyperparameters
grid={'C':np.linspace(1, 10, 10), 'kernel':['linear', 'poly','rbf', 'sigmoind']}
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
model=GridSearchCV(SVC(gamma='auto'), grid, scoring ='f1', cv=5)
model.fit(X_train_scaled, y_train)


# In[14]:


best_parameters=model.best_params_
print(best_parameters)


# In[17]:


#evaluate the model on the unseen test data
predicted_class=model.predict(X_test)
#creating a confusion matrix
from sklearn.metrics import confusion_matrix
cm=pd.DataFrame(confusion_matrix(y_test, predicted_class))
cm['Total']=np.sum(cm, axis=1)
cm=cm.append(np.sum(cm, axis=0), ignore_index=True)
cm.columns=['Predicted No', 'Predicted Yes', 'Total']
cm=cm.set_index([['Actual No', 'Actual Yes', 'Total']])
print(cm)


# In[18]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train, y_train)


# In[19]:


intercept=model.intercept_
coefficients=model.coef_
coef_list=list(coefficients[0,:])
coef_df=pd.DataFrame({'Feature': list(X_train.columns),'Coefficient': coef_list})
print(coef_df)


# In[20]:


predicted_prob=model.predict_proba(X_test)[: , 1]
predicted_class=model.predict(X_test)
from sklearn.metrics import confusion_matrix
import numpy as numpy
cm=pd.DataFrame(confusion_matrix(y_test, predicted_class))
cm['Total']=np.sum(cm, axis=1)
cm=cm.append(np.sum(cm, axis=0), ignore_index=True)
cm.columns=['Predicted No', 'Predicted Yes', 'Total']
cm=cm.set_index([["Actual", "Actual Yes", "Total"]])
print(cm)


# In[ ]:




