## import comet_ml at the top of your file
# from comet_ml import Experiment

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import baseline_model_func as utils
from sklearn import metrics



my_key = 'IGFjJ2mP1ZZPVHurdGupI5DJt'

# Create an experiment with your api key
experiment = Experiment(
    api_key=my_key,
    project_name="baseline model",
    workspace="2nd milestone",
)


seed = 100
df = pd.read_csv('baseline_model_data.csv')

df = df.rename(columns={'shot distance': 'shot_distance',
                   'shot angle': 'shot_angle',
                   'empty net': 'empty_net',
                   'is goal': 'is_goal'})

df_train, df_test = utils.separate(season_threshold=2020, df=df)
# feature_columns = ['shot_distance', 'shot_angle', 'empty_net']
feature_columns = ['shot_distance']
target_column = 'is_goal'

X = df_train[feature_columns]
y = df_train.is_goal

# see the distribution of the data for train and val
utils.distribution(y=y, name='data')

X_train, X_val, y_train, y_val = train_test_split(X, y,
                                                  random_state=seed,
                                                  test_size=0.2, stratify=y)

utils.distribution(y=y_val, name='y_val')
utils.distribution(y=y_train, name='y_train')

# we have the same distribution for our train and val as the data
logistic_classifier = LogisticRegression(random_state=seed)
logistic_classifier.fit(X_train, y_train)

y_pred = logistic_classifier.predict(X_val)

cnf_matrix = confusion_matrix(y_val, y_pred)
print(cnf_matrix)

acc = metrics.accuracy_score(y_val, y_pred)
print(acc)

utils.heatmap(cnf_matrix)

target_names = ['not_goal', 'is_goal']
y_prob = logistic_classifier.predict_proba(X_val)

report = classification_report(y_val, y_pred,
                              target_names=target_names,
                              zero_division=0.0)
print(report)


f1 = f1_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)

params = {
    "random_state": seed,
    "model_type": "logistic_regression",
    "stratify": True,
}

metrics_dictionary = {"f1": f1, "recall": recall, "precision": precision}

experiment.log_dataset_hash(X_train)
experiment.log_parameters(params)
experiment.log_metrics(metrics_dictionary)

experiment.end()

data = pd.DataFrame({'true label': y_val,
                     'prob of not a goal': y_prob[:, 0],
                     'prob of goal': y_prob[:, 1],
                     'prediction': y_pred})

data_logistic = pd.concat((X_val, data), axis=1)

#
# plt.figure(figsize=[8, 6])
# fpr, tpr, _ = data_logistic.roc_curve(y_val,  y_prob[:, 1])
# plt.plot(fpr, tpr)
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.savefig('roc_curve.png', dpi=200)
# plt.show()
# print(metrics.auc(fpr, tpr))
#
#
# # goal rate
# q = y_prob[:, 1].groupby(pd.qcut(X_val, 10))['is_goal'].mean()

