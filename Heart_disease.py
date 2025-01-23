#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[2]:


ds=pd.read_csv('heart-disease.csv')


# In[3]:


ds


# age: Age of the patient (in years)
# sex: Sex of the patient (1 = male, 0 = female)
# cp: Chest pain type (1-4)
# trestbps: Resting blood pressure (in mm Hg on admission to the hospital)
# chol: Serum cholesterol in mg/dl
# fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
# restecg: Resting electrocardiographic results (0-2)
# thalach: Maximum heart rate achieved
# exang: Exercise-induced angina (1 = yes; 0 = no)
# oldpeak: ST depression induced by exercise relative to rest
# 

# In[5]:


shuffle_ds=ds.sample(frac=1,random_state=42).reset_index(drop=True)  


# frac=1 as it ensures all the rows in a dataset are included in a shuffle,
# random_state=42 set a random seed for reproducibility

# In[7]:


shuffle_ds


# In[8]:


X=shuffle_ds.iloc[:,:-1].values


# In[9]:


X


# In[10]:


y=shuffle_ds.iloc[:,-1].values


# In[11]:


y


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[14]:


X_train.shape


# In[15]:


y_train.shape


# In[16]:


#we use model_selection to see which model give us better prediction


# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 


# In[18]:


models=[
    ('lr',LogisticRegression()), 
    ('dtc',DecisionTreeClassifier()),
    ('rfc',RandomForestClassifier()), 
    ('gbc',GradientBoostingClassifier()), 
    ('knc',KNeighborsClassifier()), 
    ('gnb',GaussianNB()),
    ('svc',SVC())
]


# In[19]:


import warnings
warnings.filterwarnings('ignore')


# In[20]:


for name,model in models:
    clf=model
    clf.fit(X_train,y_train)
    Accuracy=clf.score(X_test,y_test)
    print(name,': ',Accuracy) 


# In[21]:


Gnb=GaussianNB()


# In[22]:


Gnb.fit(X_train,y_train)


# In[23]:


Gnb.score(X_test,y_test)


# In[66]:


import pickle


# In[68]:


Gnb.predict(np.array([[51,1,2,94,227,0,1,154,1,0.0,2,1,3]]))


# In[70]:


Gnb.predict(np.array([[57,1,0,150,276,0,0,112,1,0.6,1,1,1]]))


# In[72]:


with open('model.pkl','wb') as f:
    pickle.dump(Gnb,f)


# In[ ]:




