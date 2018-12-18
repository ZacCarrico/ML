# Purpose: Explore Random Forest Classification

# reference https://chrisalbon.com/machine_learning/trees_and_forests/random_forest_classifier_example/


from __future__ import print_function
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
np.random.seed(0)

iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
print("df.head():", df.head())

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("iris.target:", iris.target)
print("iris.target_names:", iris.target_names)

df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75

train, test = df[df['is_train'] == True], df[df['is_train'] == False]

print("len(train): ", len(train))
print("len(test): ", len(test))

features = df.columns[:4]
print("features:", features)

y = pd.factorize(train['species'])[0]
print("y:", y)

clf = RandomForestClassifier(n_jobs = 2, random_state = 0)
clf.fit(train[features], y)

print("clf.predict(test[features]):", clf.predict(test[features]))

print("clf.predict(test[features]):", clf.predict(test[features]))

preds = iris.target_names[clf.predict(test[features])]

print("confusion matrix:")
print(pd.crosstab(
    test['species'], 
    preds, 
    rownames = ['Actual Species'], 
    colnames = ['Predicted Species']))

print("list(zip(train[features], clf.feature_importances_)):", list(zip(train[features], clf.feature_importances_)))