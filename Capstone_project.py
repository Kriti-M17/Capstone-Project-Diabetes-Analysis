#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


data=pd.read_csv("D:\simpli learn\Final project Capstone\health care diabetes.csv")


# In[3]:


data.head()


# In[4]:


data.isnull().any()


# In[5]:


data.info()


# In[6]:


Positive = data[data['Outcome']==1]
Positive.head(5)


# In[7]:


data['Glucose'].value_counts().head(7)


# In[8]:


plt.hist(data['Glucose'])


# In[9]:


data['BloodPressure'].value_counts().head(7)


# In[10]:


plt.hist(data['BloodPressure'])


# In[11]:


data['SkinThickness'].value_counts().head(7)


# In[12]:


plt.hist(data['SkinThickness'])


# In[13]:


data['Insulin'].value_counts().head(7)


# In[14]:


plt.hist(data['Insulin'])


# In[15]:


data['BMI'].value_counts().head(7)


# In[16]:


plt.hist(data['BMI'])


# In[17]:


data.describe().transpose()


# In[18]:


plt.hist(Positive['BMI'],histtype='stepfilled',bins=20)


# In[19]:


Positive['BMI'].value_counts().head(7)


# In[20]:


plt.hist(Positive['Glucose'],histtype='stepfilled',bins=20)


# In[21]:


Positive['Glucose'].value_counts().head(7)


# In[22]:


plt.hist(Positive['BloodPressure'],histtype='stepfilled',bins=20)


# In[23]:


Positive['BloodPressure'].value_counts().head(7)


# In[24]:


plt.hist(Positive['SkinThickness'],histtype='stepfilled',bins=20)


# In[25]:


Positive['SkinThickness'].value_counts().head(7)


# In[26]:


plt.hist(Positive['Insulin'],histtype='stepfilled',bins=20)


# In[27]:


Positive['Insulin'].value_counts().head(7)


# In[28]:


#Scatter Plot


# In[29]:


BloodPressure = Positive['BloodPressure']
Glucose = Positive['Glucose']
SkinThickness = Positive['SkinThickness']
Insulin = Positive['Insulin']
BMI = Positive['BMI']


# In[30]:


plt.scatter(BloodPressure, Glucose, color=['b'])
plt.xlabel('BloodPressure')
plt.ylabel('Glucose')
plt.title('BloodPressure & Glucose')
plt.show()


# In[31]:


g =sns.scatterplot(x= "Glucose" ,y= "BloodPressure",
              hue="Outcome",
              data=data);


# In[32]:


B =sns.scatterplot(x= "BMI" ,y= "Insulin",
              hue="Outcome",
              data=data);


# In[33]:


S =sns.scatterplot(x= "SkinThickness" ,y= "Insulin",
              hue="Outcome",
              data=data);


# In[34]:


### correlation matrix
data.corr()


# In[35]:


### create correlation heat map
sns.heatmap(data.corr())


# In[36]:


plt.subplots(figsize=(8,8))
sns.heatmap(data.corr(),annot=True,cmap='viridis')  ### gives correlation value


# In[37]:


# Logistic Regreation and model building


# In[38]:


data.head(5)


# In[39]:


features = data.iloc[:,[0,1,2,3,4,5,6,7]].values
label = data.iloc[:,8].values


# In[40]:


#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,
                                                label,
                                                test_size=0.2,
                                                random_state =10)


# In[41]:


#Create model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train) 


# In[42]:


print(model.score(X_train,y_train))
print(model.score(X_test,y_test))


# In[43]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(label,model.predict(features))
cm


# In[44]:


from sklearn.metrics import classification_report
print(classification_report(label,model.predict(features)))


# In[45]:


#Preparing ROC Curve (Receiver Operating Characteristics Curve)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# predict probabilities
probs = model.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(label, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(label, probs)
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')


# In[46]:


#Applying Decission Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier(max_depth=5)
model3.fit(X_train,y_train)


# In[47]:


model3.score(X_train,y_train)


# In[48]:


model3.score(X_test,y_test)


# In[49]:


#Applying Random Forest
from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier(n_estimators=11)
model4.fit(X_train,y_train)


# In[50]:


model4.score(X_train,y_train)


# In[51]:


model4.score(X_test,y_test)


# In[52]:


#Support Vector Classifier

from sklearn.svm import SVC 
model5 = SVC(kernel='rbf',
           gamma='auto')
model5


# In[53]:


model5.fit(X_train,y_train)


# In[54]:


model5.score(X_test,y_test)


# In[55]:


model5.score(X_train,y_train)


# In[56]:


#Applying K-NN
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=7,
                             metric='minkowski',
                             p = 2)
model2.fit(X_train,y_train)


# In[57]:


#Preparing ROC Curve (Receiver Operating Characteristics Curve)
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# predict probabilities
probs = model2.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# calculate AUC
auc = roc_auc_score(label, probs)
print('AUC: %.3f' % auc)
# calculate roc curve
fpr, tpr, thresholds = roc_curve(label, probs)
print("True Positive Rate - {}, False Positive Rate - {} Thresholds - {}".format(tpr,fpr,thresholds))
# plot no skill
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")


# In[58]:


#Precision Recall Curve for Logistic Regression

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.')


# In[59]:


#Precision Recall Curve for KNN

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model2.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model2.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.')


# In[60]:


#Precision Recall Curve for Decission Tree Classifier

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model3.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model3.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.')


# In[61]:


#Precision Recall Curve for Random Forest

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
# predict probabilities
probs = model4.predict_proba(features)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
# predict class values
yhat = model4.predict(features)
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(label, probs)
# calculate F1 score
f1 = f1_score(label, yhat)
# calculate precision-recall AUC
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
# plot no skill
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
# plot the precision-recall curve for the model
plt.plot(recall, precision, marker='.')


# In[ ]:




