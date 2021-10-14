"""Iris Flower Type Prediction
Program is part of Simplearn Machine Learning Course
Date:14 October 2021
Done by: Sofien Abidi"""

#Loading Libraries & Datasets
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#Setting random seed
np.random.seed(0)

#Orginize iris data into dataframe
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns = iris.feature_names)
df_iris['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)


#Select random trianing/testing data
df_iris['is_train'] = np.random.uniform(0, 1, len(df_iris))<.75
df_iris.head()

#Splitting traing and testing dataframe
train = df_iris[df_iris['is_train'] == True]
test = df_iris[df_iris['is_train'] == False]

#Defining the features columns
feature = df_iris.columns[:4]

#Converting species names into digits
y_train = pd.factorize(train['species'])[0]
y_test = pd.factorize(test['species'])[0]

#Model creation and training
clf = RandomForestClassifier(n_jobs=2,random_state=0)
clf.fit(train[feature],y_train)

#Model Testing
y_pred = clf.predict(test[feature])
from sklearn.metrics import accuracy_score, confusion_matrix
score = accuracy_score(y_pred,y_test)
cf = confusion_matrix(y_pred,y_test)
print(score)
print(cf)
