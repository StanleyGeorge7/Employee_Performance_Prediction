#!/usr/bin/env python
# coding: utf-8

# #### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.svm import SVC
from statistics import mean
import warnings
import pickle
from sklearn import metrics
import plotly.express as px
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


# #### Importing Dataset and Initial Analysis on Data

# In[2]:


df = pd.read_excel(r'D:\bepec\coding\Assignment\Data Analytics Project\INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls')
df.head()


# In[3]:


df.drop(labels='EmpNumber',axis=1,inplace=True)


# In[4]:


df.describe()


# In[5]:


df.info()


# #### Check for the Presence of Null Values

# In[6]:


df.isnull().sum()


# #### Converting all Object column to Numerical columns

# In[7]:


encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtypes=='O':
        df[col]=encoder.fit_transform(df[col])
        df[col]= df[col].astype('int64') ##to keep uniform datatype int64 instead of int32


# In[8]:


X=df.iloc[:,:-1] 
y=df.iloc[:,-1]


# #### Feature Selection

# In[9]:


# RFECV
from sklearn.feature_selection import RFECV
rfe_cv = RFECV(estimator=RandomForestClassifier(),step=1,cv=5,scoring='f1_weighted')
rfecv_fit = rfe_cv.fit(X,y)


# In[10]:


metrics.SCORERS.keys()


# In[11]:


x = rfe_cv.transform(X)
columns=list(X.columns[rfe_cv.support_])
#After feature selection
X= pd.DataFrame(x,columns=columns)
X

##scikit-image==0.14.5cls
# ### Since there is an inbalance dataset we need to handle it

# In[12]:

from imblearn.over_sampling import SMOTE
smote=SMOTE(sampling_strategy='all')
X_sm,y_sm = smote.fit_resample(X,y)


# #### Decision Tree

# In[15]:


X_train,X_test,y_train,y_test = train_test_split(X_sm,y_sm,test_size=0.3,random_state=5,stratify=y_sm)


# In[16]:


model=DecisionTreeClassifier()
model.fit(X_train,y_train)


# In[17]:


pred = model.predict(X_test)


# In[18]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[19]:


accuracy_score(y_test,pred)*100


# In[20]:


print(classification_report(y_test,pred))


# In[21]:


pd.DataFrame(
    zip(X_train.columns, abs(model.feature_importances_)),
    columns=["feature", "weight"],
).sort_values("weight").reset_index(drop=True)


# #### Hyperparameter tuning for decision tree

# In[22]:


tree_param = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150],
             'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
clf = GridSearchCV(DecisionTreeClassifier(), tree_param, cv=10)
clf.fit(X_train, y_train)


# In[23]:


clf.best_estimator_


# In[24]:


X_train,X_test,y_train,y_test = train_test_split(X_sm,y_sm,test_size=0.3,random_state=2,stratify=y_sm)
model_tuned_decision_tree=DecisionTreeClassifier(max_depth=11,max_leaf_nodes=32,min_samples_split=3)
model_tuned_decision_tree.fit(X_train,y_train)
pred = model_tuned_decision_tree.predict(X_test)
print('Accuracy of decision tree after tuning',round(accuracy_score(y_test,pred)*100,2),'%')


# In[25]:


print(classification_report(y_test,pred))


# #### RandomForest Classifier

# In[26]:


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
accuracy_rfc= []
model_rfc = RandomForestClassifier()
  
for train_index, test_index in skf.split(X_sm, y_sm):
    x_train_fold, x_test_fold = X_sm.loc[train_index], X_sm.loc[test_index]
    y_train_fold, y_test_fold = y_sm.loc[train_index], y_sm.loc[test_index]
    model_rfc.fit(x_train_fold, y_train_fold)
    accuracy_rfc.append(model_rfc.score(x_test_fold, y_test_fold))


print('Accuracy of Random Forest',round(mean(accuracy_rfc)*100,2),'%')


# #### SVM

# In[27]:


skf_svm = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)
accuracy_svc= []
model_svc = SVC()
  
for train_index, test_index in skf_svm.split(X_sm, y_sm):
    x_train_fold, x_test_fold = X_sm.loc[train_index], X_sm.loc[test_index]
    y_train_fold, y_test_fold = y_sm.loc[train_index], y_sm.loc[test_index]
    model_svc.fit(x_train_fold, y_train_fold)
    accuracy_svc.append(model_svc.score(x_test_fold, y_test_fold))


print('Accuracy of SVM Before tuning',round(mean(accuracy_svc)*100,2),'%')


# #### Hyper parameter tuning for svm

# In[28]:


params= {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}


# In[29]:


X_train,X_test,y_train,y_test = train_test_split(X_sm,y_sm,test_size=0.3,random_state=2,stratify=y_sm)
clf =GridSearchCV(SVC(),param_grid=params,cv=skf_svm)
clf.fit(X_train,y_train)


# In[30]:


clf.best_estimator_


# #### After tuning

# In[31]:


skf_svm = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
accuracy_svc= []
model_svc_tuned = SVC(C=100, gamma=0.01)
  
for train_index, test_index in skf_svm.split(X_sm, y_sm):
    x_train_fold, x_test_fold = X_sm.loc[train_index], X_sm.loc[test_index]
    y_train_fold, y_test_fold = y_sm.loc[train_index], y_sm.loc[test_index]
    model_svc_tuned.fit(x_train_fold, y_train_fold)
    accuracy_svc.append(model_svc_tuned.score(x_test_fold, y_test_fold))


print('Accuracy of SVM Before tuning',round(mean(accuracy_svc)*100,2),'%')


# #### Voting Classifier (Ensemble Classifer)

# In[32]:


Classifier=VotingClassifier(estimators=[('dt',model_tuned_decision_tree),('rfe',model_rfc),('svc',model_svc_tuned)],voting='hard')
Classifier.fit(X_train,y_train)
prediction= Classifier.predict(X_test)
#printing accuracy of the model
print("Voting Classifier Overall Accuracy = {0:.2f}%".format(accuracy_score(y_test, prediction)*100))


# In[34]:


skf_vote = StratifiedKFold(n_splits=100, shuffle=False)
Classifier=VotingClassifier(estimators=[('dt',model_tuned_decision_tree),('rfe',model_rfc),('svc',model_svc_tuned)],voting='hard')
accuracy_vote= []
for train_index, test_index in skf_vote.split(X_sm, y_sm):
    x_train_fold, x_test_fold = X_sm.loc[train_index], X_sm.loc[test_index]
    y_train_fold, y_test_fold = y_sm.loc[train_index], y_sm.loc[test_index]
    Classifier.fit(x_train_fold, y_train_fold)
    accuracy_vote.append(model_svc_tuned.score(x_test_fold, y_test_fold))
print('Accuracy of Voting classifier',round(mean(accuracy_vote)*100,2),'%')


# #### Evaluation Metrics

# In[35]:


cm =confusion_matrix(y_test,prediction)
cm_df = pd.DataFrame(cm,index=['Low','Medium','High'],columns=['Low','Medium','High'])
fig= px.imshow(cm_df,text_auto=True,labels=dict(x="Predicted Value", y="Actual Value"))
fig.update_xaxes(side="top")
fig.show()


# In[36]:


print(classification_report(y_test,prediction))


