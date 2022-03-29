#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as plt
import xgboost as xgb
from matplotlib import pyplot


# In this work was used the XGBoost to preview if a patient, who was afected for VAERS from Covid 19 vaccine, was recovered or not This was based in the patient's symptoms.

# Let's begin knowing the data from patient, vaccine (vax) and symptoms. 

# In[ ]:


patient=pd.read_csv("../input/covid19-vaccine-adverse-reactions/2021VAERSDATA.csv", encoding = 'unicode_escape', engine ='python')
patient.head(3).T


# In[ ]:


vax=pd.read_csv("../input/covid19-vaccine-adverse-reactions/2021VAERSVAX.csv")
vax.head(3).T


# In[ ]:


symptoms=pd.read_csv("../input/covid19-vaccine-adverse-reactions/2021VAERSSYMPTOMS.csv")
symptoms.head(3).T


# Let's merging patient+vax+symptoms = df2

# In[ ]:



df1 = pd.merge(vax, patient, left_on='VAERS_ID', right_on='VAERS_ID')
df2 = pd.merge(df1, symptoms, left_on='VAERS_ID', right_on='VAERS_ID')
print(df2.shape)
df2[0:3].T


# To this work will be referred only to a Covid 19 cases (df3):

# In[ ]:


df3=df2.loc[    df2['VAX_TYPE']=='COVID19'    ]
df3.head(3).T


# Checking that only Covid 19 is on

# In[ ]:



df3['VAX_TYPE'].value_counts()


# Extracting from the dataframe only the desired columns for this project

# In[ ]:


df4=df3[  ['VAERS_ID','AGE_YRS', 'SEX','SYMPTOM1','SYMPTOM2','SYMPTOM3','SYMPTOM4','SYMPTOM5','VAX_MANU','RECOVD']  ] 
df4.head(8).T


# Checking the output of project: RECOVD column

# In[ ]:


df4['RECOVD'].value_counts()


# Unknown values of RECOVD (U) are not interest to the project, so let's delete U (unknown) data in RECOVD 

# In[ ]:


df4["RECOVD"].replace ('U', np.nan, inplace=True)

#Drop rows NaN in RECOVD
df4.dropna(subset=["RECOVD"], axis=0, inplace=True)

#Reset index,cause we droped any rows
df4.reset_index(drop=True, inplace=True)


# Checking..

# In[ ]:


print('Quantity of nulls values= ',df4['RECOVD'].isnull().sum())
df4.head(10).T


# Let's avaliate the SYMPTOM1 case, as example

# In[ ]:


df4['SYMPTOM1'].value_counts()[:50]
df4_s1=pd.DataFrame(df4['SYMPTOM1'].value_counts()[:50])
df4_s1


# Let's define a new column, AC, as accumulated values of SYMPTOM1 quantity

# In[ ]:



df4_s1['AC']=df4_s1['SYMPTOM1']
for k in range (1,50):
    df4_s1['AC'][k]=(   (df4_s1['AC'][k-1]) +  (df4_s1['SYMPTOM1'][k])  )
df4_s1.head()


# Now let's define a new column as percentage of AC (%)

# In[ ]:


total=np.sum(df4_s1['SYMPTOM1'])
df4_s1['%']= df4_s1['AC']/total*100
df4_s1.head(50)


# Note that only first ten symptoms are representing 54% of the total (50)symptoms. Take a look at bar chart.

# In[ ]:


df4_s1.plot.bar(y=['%'], alpha=0.8, figsize=(12,6)) 
#plt.title('Symptom 1', size=24)
#plt.ylabel('%')


# Let's to know each of the ten symptoms' classes.
# We'l work with only ten first symptoms from SYMPTOM1 to SYMPTOM5

# In[ ]:


print(df4['SYMPTOM1'].value_counts()[:10])
print(df4['SYMPTOM2'].value_counts()[:10])
print(df4['SYMPTOM3'].value_counts()[:10])
print(df4['SYMPTOM4'].value_counts()[:10])
print(df4['SYMPTOM5'].value_counts()[:10])


# Create a new dataframe to preserve the data from df4 

# In[ ]:


df5=df4


# Now let's create a function to select only the lines that contain the ten main symptoms.
# The remaining ones will be overwrite as "Others"

# In[ ]:


def Purge (list,column):
    k=0
    while (k<len(df5)):
        for i in list:
            if  df5[column][k] not in list1:
                df5[column][k]='Others' 
        k=k+1
    return ()


# Let's define the parameters to the Purge function:

# In[ ]:


#list1
list1=df5['SYMPTOM1'].value_counts()[:10]
list1=list1.index
#column1
column1='SYMPTOM1'

#list2
list2=df5['SYMPTOM2'].value_counts()[:10]
list2=list2.index
#column2
column2='SYMPTOM2'

#list3
list3=df5['SYMPTOM3'].value_counts()[:10]
list3=list3.index
#column3
column3='SYMPTOM3'

#list4
list4=df5['SYMPTOM4'].value_counts()[:10]
list4=list4.index
#column4
column4='SYMPTOM4'

#list5
list5=df5['SYMPTOM5'].value_counts()[:10]
list5=list5.index
#column5
column5='SYMPTOM5'


# Let's perform the Purge function to each SYMPTOM (it will demand any minutes)

# In[ ]:


Purge(list1,column1)
print('SYMPTOM1 - done')
Purge(list2,column2)
print('SYMPTOM2 - done')
Purge(list3,column3)
print('SYMPTOM3 - done')
Purge(list4,column4)
print('SYMPTOM4 - done')
Purge(list5,column5)
print('SYMPTOM5 - done')


# Let's take a overview at the new dataframe

# In[ ]:


df5.head()


# Now, let's encode the df5

# In[ ]:


from sklearn.preprocessing import LabelEncoder
en = LabelEncoder()
columns=['SEX','SYMPTOM1','SYMPTOM2','SYMPTOM3','SYMPTOM4','SYMPTOM5','VAX_MANU','RECOVD'] 
for i in columns:
    df5[i] = en.fit_transform(df5[i])
    
df5.head()  


# Defining train set and test set

# In[ ]:


Y = df5["RECOVD"]
y=pd.DataFrame(Y)
X = df5.drop("RECOVD",axis = 1)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# Instantiate the XGBClassifier: xg_cl

# In[ ]:


xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)


# Fit the classifier to the training set

# In[ ]:


xg_cl.fit(X_train,y_train)


# Predict the labels of the test set: preds

# In[ ]:


preds = xg_cl.predict(X_test)
preds=preds.reshape(7299,1)


# Compute the accuracy: accuracy

# In[ ]:


accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))


# The accuracy is not good enough. 
# Perhaps if we work with more than ten symptoms the result could be better.

# I hope that work could be useful to continue studies of VAERS from Covid 19 vaccines.
# Thank you for your up vote.

# In[ ]:




