#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
#%matplotlib inline

filepath = "/Users/kenny/PythonML/Titanic/"

data = pd.read_csv(filepath + "train.csv")


# In[2]:


for col in data.columns:
    print("{}:{}".format(col,data[col].isnull().sum()))


# In[3]:


#data['Age'].hist()
data['Pclass'].hist()


data['Age'] = data['Age'].fillna(data.groupby('Sex')['Age'].transform('mean'))

for col in data.columns:
    print("{}:{}".format(col,data[col].isnull().sum()))
    
data = data.drop(["PassengerId",'Name','Cabin','Ticket'], axis=1)
print(data)


# In[4]:


#one hot encoding

Pclass_encode = pd.get_dummies(data['Pclass'])
# Drop column Pclass as it is now encoded
data = data.drop('Pclass',axis = 1)
# Join the encoded df
data = data.join(Pclass_encode)

Embarked_encode = pd.get_dummies(data['Embarked'])
# Drop column Pclass as it is now encoded
data = data.drop('Embarked',axis = 1)
# Join the encoded df
data = data.join(Embarked_encode)


# In[5]:


data['Sex'].replace(to_replace =["male"], value =1, inplace = True) 
data['Sex'].replace(to_replace =["female"], value =0, inplace = True) 


# In[6]:


#import libraries
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn as sns
from matplotlib import pyplot

data.head(20)


# In[7]:


##split test train
# where y is target and X is features

y = data['Survived']

X = data.drop(['Survived'],axis=1)

print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=42)


# In[ ]:





# In[8]:


##training model 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k='all')
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(12,'Score'))  #print 10 best features


# In[9]:


# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=20, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


# In[13]:


model = LogisticRegression(solver='liblinear', multi_class='ovr')
model.fit(X_train, y_train)
predictions = model.predict(X_test)


# In[14]:


# Evaluate predictions
print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


# In[ ]:




