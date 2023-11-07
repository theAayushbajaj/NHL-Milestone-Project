import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import baseline_model_func as utils

seed = 100
df = pd.read_csv('baseline_model_data.csv')
df_train, df_test = utils.separate(season_threshold=2020, df=df)
# feature_columns = ['shot distance', 'shot angle', 'empty net']
feature_columns = ['shot distance']
target_column = 'is goal'

X = df_train[feature_columns]
y = df_train[target_column]

# see the distribution of the data for train and val
utils.distribution(y=y, name='data')

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=seed, test_size=0.2)

utils.distribution(y=y_val, name='y_val')
utils.distribution(y=y_train, name='y_train')

# we have the same distribution for our train and val as the data
logistic_classifier = LogisticRegression(random_state=seed)
logistic_classifier.fit(X_train, y_train)

y_pred = logistic_classifier.predict(X_val)

cnf_matrix = confusion_matrix(y_val, y_pred)
utils.heatmap(cnf_matrix)

target_names = ['not a goal', 'is a goal']
y_prob = logistic_classifier.predict_proba(X_val)

print(classification_report(y_val, y_pred, target_names=target_names, zero_division='warn'))

data = pd.DataFrame({'true label': y_val,
                     'prob of not a goal': y_prob[:, 0],
                     'prob of goal': y_prob[:, 1],
                     'prediction': y_pred})

data_logistic = pd.concat((X_val, data), axis=1)