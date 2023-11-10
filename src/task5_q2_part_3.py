from comet_ml import Experiment
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Sckit learn modules and classes
from sklearn.metrics import confusion_matrix, f1_score, \
precision_score, recall_score, classification_report, accuracy_score, \
roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
# encoding
import category_encoders as ce
import xgboost as xgb
import os
import warnings
import utils_ar_pa as ut

seed = 200

######################################
my_key = os.environ.get("comet_key")
print('Experiment Started!')
# Create an experiment with your api key
experiment = Experiment(
    api_key=my_key,
    project_name="baseline model",
    workspace="2nd milestone",
    log_code=True,
    auto_param_logging=True,
    auto_metric_logging=True)
######################################
# in this part we will train the xgboost classifier

# first let's read the Train, Test, Val dataframes taken from part_2
Train = pd.read_csv('Train.csv', index_col=0)
Val = pd.read_csv('Val.csv', index_col=0)
Test = pd.read_csv('Test.csv', index_col=0)



# we train a model without tuning it
xgb_clf = xgb.XGBClassifier( objective='binary:logistic',
                               n_estimators=400, max_depth=5, learning_rate=0.1, subsample=0.7,
                               colsample_bytree=1,  reg_lambda=0.65,
                               seed=seed)

X_train = Train.iloc[:, :-1]
X_val = Val.iloc[:, :-1]
y_train = Train.iloc[:, -1]
y_val = Val.iloc[:, -1]

xgb_clf.fit(X_train, y_train)

y_pred = xgb_clf.predict(X_val)
y_prob = xgb_clf.predict_proba(X_val)

results = pd.DataFrame({'is_goal': y_val, 'is_goal_pred': y_pred, 'prop_goal': y_prob[:, 1],
                       'prob_not_goal': y_prob[:, 0]})

print(results)
results.to_csv('adv_prob_q2_no_tune.csv')

cfn_mat = confusion_matrix(y_val, y_pred)
print(cfn_mat)

labels = ['not_goal', 'is_goal']
sns.heatmap(cfn_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            cbar=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("adv_prob_no_tune_confusion_matrix.png")
plt.show()


acc = accuracy_score(y_val, y_pred)
fsc = f1_score(y_val, y_pred)
psc = precision_score(y_val, y_pred)
rsc = recall_score(y_val, y_pred)
balanced_acc = balanced_accuracy_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

print(f'accuracy {acc}')
print(f'f1_score {fsc}')
print(f'precision_score {psc}')
print(f'recall_score {rsc}')
print(f'balanced_accuracy_score {balanced_acc}')
print(f'area under the curve {auc}')

print(report)

model_filename = 'xgboost_model_q2_no_tune.json'
xgb_clf.save_model(model_filename)

features = Train.columns.to_list()[:-1]
target = 'is_goal'
model_filename = 'xgboost_model_q2_no_tune.json'
xgb_clf.save_model(model_filename)

model = xgb.XGBClassifier()
model.load_model(model_filename)
name = 'xgboost_q2'
ut.plot_calibrations_task5(model, features, target, Val, name)

features = Train.columns.to_list()[:-1]
target = 'is_goal'

model = xgb.XGBClassifier()
model.load_model(model_filename)
name = 'xgboost_q1_no_tune'
ut.plot_calibrations_task5(model, features, target, Val, name)

parameters_log = {"n_estimators":400, "max_depth":5, "learning_rate":0.1,
                  "subsample":0.7, "colsample_bytree":1.0,  "reg_lambda":0.65, "seed":seed}

metrics_log = {"accuracy": acc, "f1_score": fsc,
               "precision_score":psc, "balanced_accuracy_score":balanced_acc,
               "recall_score": rsc, "area under the curve":auc}

Train.to_csv()
experiment.log_table("Train_norm.csv", Train)
experiment.log_table("Val_norm.csv", Val)
experiment.log_table("Test_norm.csv", Test)
experiment.log_table("confusion_matrix_q1_no_tune.csv", cfn_mat.tolist())
experiment.log_table("probabilities_no_tune.csv", results)

experiment.log_parameters(parameters_log)
experiment.log_metrics(metrics_log)
experiment.log_model(xgb_clf, model_filename, overwrite=True)
experiment.log_image("adv_prob_no_tune_confusion_matrix.png", "Confusion_matrix")
experiment.log_image("dataset_balance.png", "Dataset unbalanced")
experiment.log_image("xgboost_q1_no_tune_diagrams.png", "Diagrams")

experiment.log_confusion_matrix(y_val, y_pred,
    title = "Confusion Matrix: Evaluation",
    file_name = 'xgboost_q1_no_tune_diagrams.png')

experiment.end()
print('Experiment Ended!')
