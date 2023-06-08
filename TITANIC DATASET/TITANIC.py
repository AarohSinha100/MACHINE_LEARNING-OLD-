#!/usr/bin/env python
# coding: utf-8

# In[157]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### BECOMING ONE WITH THE DATA

# In[158]:


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


# In[159]:


train_df.head()


# In[160]:


test_df.head()


# In[161]:


# Checking the number of passengers
print(f"The number of passengers is {len(train_df)}")


# In[162]:


train_df.info()


# ### Analyzing the data

# In[163]:


for i in range(len(train_df.columns)):
    value_counts = train_df[train_df.columns[i]].value_counts().sum
    print(train_df.columns[i])
    print(value_counts)
    print("-----------------------")


# In[164]:


survived = (train_df["Survived"]==0).sum()
print(f"The number of passengers who survived are {survived}")
print(f"The number of passengers who died are {len(train_df) - survived}")


# In[165]:


# Countplot to see how many survived
sns.countplot(x="Survived",data=train_df)
#0 - did not survive
#1 - survived


# #### GENDER

# In[166]:


male = (train_df["Sex"] == "male").sum()
print(f"The number of males - {male}")
print(f"The number of females - {len(train_df)-male}")


# In[167]:


sns.countplot(x="Survived",hue="Sex",data=train_df)


# In[168]:


train_df["Survived"].iloc[2]


# In[169]:


male_survived = 0
female_survived = 0
for i in range(len(train_df)):
    if train_df["Sex"].iloc[i]=="male":
        if train_df["Survived"].iloc[i] == 1:
            male_survived+=1
    elif train_df["Sex"].iloc[i]=="female":
        if train_df["Survived"].iloc[i] == 1:
            female_survived+=1 


# In[170]:


print(f"Total {male_survived} males out of {male} survived")
print(f"Total {female_survived} females survived out of {len(train_df) - male}")


# #### PClass

# In[171]:


train_df["Pclass"].value_counts()


# In[172]:


sns.countplot(x="Survived",hue="Pclass",data=train_df)


# #### AGE

# In[173]:


train_df["Age"].plot.hist()
#There were more 10-40 year old passengers travelling


# In[174]:


# replacing Nan values by mean
train_df["Age"].isnull().sum()


# In[175]:


train_df["Age"] = train_df["Age"].replace(0, np.NaN)
mean = int(train_df["Age"].mean(skipna=True))
train_df["Age"] = train_df["Age"].replace(np.NaN, mean)
        
    


# In[176]:


train_df["Age"].isnull().sum()


# In[177]:


train_df["Age"].plot.hist()


# In[178]:


sns.boxplot(x="Pclass",y="Age",data=train_df)


# In[179]:


train_df["SibSp"].value_counts() #Number of siblings and spouse on board


# In[180]:


train_df["Parch"].value_counts() #Number of parents children pairs


# In[181]:


#Embarked - Embarked implies where the traveler mounted from. 
#There are three possible values for Embark — Southampton, Cherbourg, and Queenstown.

plt.figure(figsize=(15,7))

#fig = plt.figure(figsize=(15,7))
#fig.add_subplot(2,1,2)
plt.subplot(1,2,1)
sns.countplot(x="Embarked",data=train_df)

plt.subplot(1,2,2)
sns.countplot(x="Embarked",hue="Survived",data=train_df)


# ### PreProcessing The Data

# In[182]:


train_df.head()


# In[183]:


train_df.drop("Name",axis=1,inplace=True)


# In[184]:


train_df.head(1)


# In[185]:


train_df.drop("Ticket",axis=1,inplace=True)


# In[186]:


train_df["Fare"].describe() #We will standardise this


# In[187]:


train_df["Cabin"].value_counts()


# In[188]:


train_df.drop("Cabin",axis=1,inplace=True)


# In[189]:


train_df.head(2)


# In[190]:


train_df["Sex"].isnull().sum()


# In[191]:


def encode_gender(value):
    if value=="male":
        return 1
    elif value=="female":
        return 0
    
train_df["Sex"] = train_df["Sex"].apply(lambda x: encode_gender(x))
train_df.head(5)


# In[192]:


train_df["Embarked"].value_counts()


# In[193]:


embarked = pd.get_dummies(train_df["Embarked"],drop_first=True)
embarked.head()


# In[194]:


train_df["Pclass"].value_counts()


# In[195]:


Pclass = pd.get_dummies(train_df["Pclass"])
Pclass.head()


# In[196]:


train_df = pd.concat([train_df, embarked, Pclass],axis=1)
train_df.head()


# In[197]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_df[["Age","Fare"]])


# In[198]:


scaled = scaler.transform(train_df[["Age","Fare"]])
scaled = pd.DataFrame(scaled,columns=["Age","Fare"])
scaled.head()


# In[199]:


train_df.head()


# In[200]:


train_df.drop(["Age","Embarked","Pclass","Fare"],axis=1,inplace=True)
train_df.head()


# In[201]:


train_df = pd.concat([train_df, scaled],axis=1)
train_df.head()


# ### TRAIN TEST SPLIT

# In[226]:


from sklearn.model_selection import train_test_split

#Shuffling our data
train_df_shuffled = train_df.sample(frac=1, random_state=42)

#Traintestsplit
X = train_df_shuffled.drop("Survived",axis=1)
y = train_df_shuffled["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X ,y ,test_size=0.3 ,random_state=42)


# In[227]:


len(X_train), len(X_test), len(y_train), len(y_test)


# ### Creating helper functions

# In[228]:


# Helper function to calculate all results
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def calculate_results(y_true, y_pred):
    model_accuracy = accuracy_score(y_true, y_pred) *100
    # Calculate model precision, recall and f1 score using "weighted" average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
    return model_results


# # MODEL 1 - LOGISTIC REGRESSION
# 
# Logistic regression estimates the relationship between a dependent variable and one or more independent variables and predicts a categorical variable versus a continuous one.

# In[229]:


from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression()
log_model.fit(X_train, y_train)


# In[230]:


log_preds = log_model.predict(X_test)
log_preds[:10]


# In[231]:


models = []
models.append("logistic_regression_results")
models


# In[232]:


logistic_regression_results = calculate_results(y_test, log_preds)
logistic_regression_results


# # MODEL 2: DECISSION TREE MODEL

# In[233]:


from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()


# In[234]:


dtree.fit(X_train, y_train)
dtree_preds = dtree.predict(X_test)
dtree_preds[:10]


# In[236]:


models.append("decision_tree_results")
decision_tree_results = calculate_results(y_test, dtree_preds)
decision_tree_results


# #### HYPERPARAMETER_TUNING -  N_ESTIMATORS AND MAX DEPTH

# In[247]:


import warnings
warnings.filterwarnings('ignore')

# MODEL 3: RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

#Finding the right estimators
n_estimators = [1,2,4,8,16,32,64,128,256,512]
train_results = []
test_results = []

for n_estimator in n_estimators:
    random_forest_model = RandomForestClassifier(n_estimators= n_estimator, n_jobs= -1)
    random_forest_model.fit(X_train, y_train)
    
    train_pred = random_forest_model.predict(X_train)
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    
    y_pred = random_forest_model.predict(X_test)
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
    
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, "b", label="Train AUC")
line2, = plt.plot(n_estimators, test_results, "r", label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("n_estimators")
plt.show()


# In[265]:


max_depths = np.linspace(1, 32, 3, endpoint=True)
train_results = []
test_results = []

for max_depth in max_depths:
    rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
    rf.fit(X_train, y_train) 
    
    train_pred = rf.predict(X_train)
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    
    train_results.append(roc_auc)
    
    y_pred = rf.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

    
    from matplotlib.legend_handler import HandlerLine2D
line1,= plt.plot(max_depths, train_results, "b",label= "Train AUC")
line2,= plt.plot(max_depths, test_results, "r" ,label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("Tree depth")
plt.show()


# In[273]:


random_forest_model = RandomForestClassifier(n_estimators=40, n_jobs=-1,max_depth=4)
random_forest_model.fit(X_train, y_train)

random_forest_preds = random_forest_model.predict(X_test)
random_forest_preds[:10]


# In[274]:


models.append("random_forest_model_results")
random_forest_results = calculate_results(y_test, random_forest_preds)
random_forest_results


# In[277]:


models = models[:3]
models


# In[278]:


random_forest_results


# In[282]:


## Making a Dataset from these 
results = pd.DataFrame({"Logistic_Regression":logistic_regression_results,
                       "Decision Tree":decision_tree_results,
                       "Random Forest":random_forest_results})

results = results.transpose()
results.head()


# ### Getting Some result visualizations

# In[283]:


results["accuracy"] = results["accuracy"]/100
results.head(2)


# In[287]:


results.plot(kind="bar",figsize=(7,5)).legend(bbox_to_anchor=(1.0,1.0))


# In[ ]:




