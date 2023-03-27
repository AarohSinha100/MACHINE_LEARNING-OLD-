import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import math


df = pd.read_csv("diabetes - Copy.csv")
df.head(5)

# Some Columns cannot have 0 values or they will screw our results
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for column in zero_not_accepted:
    df[column] = df[column].replace(0, np.NaN)
    mean = int(df[column].mean(skipna=True))
    df[column] = df[column].replace(np.NaN, mean)

# we replace NaN by mean because mean is the average of what a person may have, as a person cannot have 0 or NaN BloodPressure, he might be dead:)

#split dataset
X = df.iloc[:,0:8] #all the rows and column 0 to 6
y = df.iloc[:,8] #just column 8 (our result)
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0,test_size=0.2)

#feature scaling - scaling data to a range
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#value of k (must be odd)
k = math.floor(math.sqrt(len(y_test))) - 1

#defining the model
classifier = KNeighborsClassifier(n_neighbors=k,p=1,metric='euclidean')

# Fit Model
classifier.fit(X_train, y_train)

#predict the test set result
y_pred = classifier.predict(X_test)

#evaluate the model - confusion metrics
cm = confusion_matrix(y_test,y_pred)
print(cm)
print(f1_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
