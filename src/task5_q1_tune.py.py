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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# encoding
# from category_encoders import TargetEncoder
import xgboost as xgb
import os
import warnings
import utils_ar_pa as ut
# hide warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings("default")

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


# Set display options to show all columns
pd.set_option('display.max_columns', None)
# set seed
seed = 200

df_base = pd.read_csv('baseline_model_data.csv')

def renamer(df):
    df = df.copy()
    cols = df.columns.to_list()
    cols_space = [col.replace(' ', '_') for col in cols]
    cols_rename = {cols[i]: cols_space[i] for i in range(len(cols))}
    return cols_rename

# adding space to the white space in column names
cols_rename_base = renamer(df_base)
df_base.rename(columns=cols_rename_base, inplace=True)
print(df_base.columns)

# filter season 2020 for test and the remaining goes to train
df_base_test = df_base[df_base.season == 2020]
df_base_train = df_base[df_base.season < 2020]

# cols for question 1
cols = ['shot_distance', 'shot_angle']

# features
df_base_train_X = df_base_train.loc[:, cols]
df_base_test_X = df_base_test.loc[:, cols]
# targets
df_base_train_y = df_base_train.iloc[:, -1]
df_base_test_y = df_base_test.iloc[:, -1]

value_counts = df_base_train.is_goal.value_counts()
sns.barplot(x=value_counts.index, y=value_counts.values)
plt.savefig('dataset_balance.png')

X_train, X_val, y_train, y_val = train_test_split(df_base_train_X,
                                                    df_base_train_y,
                                                    random_state=seed,
                                                    test_size=0.2,
                                                    stratify=df_base_train_y)

# normalize the data for angle and shot distance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(df_base_test_X)



xgb_clf_param_grid = {'learning_rate': [0.1], 'colsample_bytree': [1.0],
                     'n_estimators': [400, 450], 'max_depth': [5, 6],
                      'reg_lambda':[0.65, 0.7], 'subsample':[0.7]}

xgb_clf = xgb.XGBClassifier(objective='binary:logistic')

grid_search = GridSearchCV(param_grid=xgb_clf_param_grid,
                            estimator=xgb_clf,
                            scoring="precision",
                            cv=5,
                            verbose=1) # verbose=1 for not showing the results
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
print("Best roc_auc found: ", grid_search.best_score_)

best_params = grid_search.best_params_
best_xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', **best_params, seed=seed)
best_xgb_classifier.fit(X_train, y_train)

y_pred = best_xgb_classifier.predict(X_val)
y_prob = best_xgb_classifier.predict_proba(X_val)

results_tuned = pd.DataFrame({'is_goal': y_val, 'is_goal_pred': y_pred, 'prop_goal': y_prob[:, 1],
                       'prob_not_goal': y_prob[:, 0]})


results = pd.DataFrame({'is_goal': y_val, 'is_goal_pred': y_pred, 'prop_goal': y_prob[:, 1],
                       'prob_not_goal': y_prob[:, 0]})

print(results)
results.to_csv('adv_prob_no_tune.csv')

cfn_mat = confusion_matrix(y_val, y_pred)
print(cfn_mat)

labels = ['not_goal', 'is_goal']
sns.heatmap(cfn_mat, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            cbar=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("adv_prob_q1_tune_confusion_matrix.png")
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


######################################
model_filename = 'xgboost_model_q1_tune.model'
best_xgb_classifier.save_model(model_filename)

desired_columns = ['season', 'shot_distance', 'shot_angle', 'is_goal']
# rebuild the dataframe for Train
y_train_df = pd.DataFrame({'is_goal': y_train.to_numpy()}).reset_index(drop=True)
X_train_df = pd.DataFrame({'shot_distance': X_train[:, 0] , 'shot_angle':X_train[:, 1]})
Train = pd.concat([X_train_df.reset_index(drop=True), y_train_df], axis=1, ignore_index=True)
Train.columns = desired_columns[1:]
# rebuild the dataframe for Val
y_val_df = pd.DataFrame({'is_goal': y_val.to_numpy()}).reset_index(drop=True)
X_val_df = pd.DataFrame({'shot_distance': X_val[:, 0] , 'shot_angle':X_val[:, 1]})
Val = pd.concat([X_val_df.reset_index(drop=True), y_val_df], axis=1, ignore_index=True)
Val.columns = desired_columns[1:]

# rebuild the dataframe for Test
y_test_df = pd.DataFrame({'is_goal': df_base_test_y.to_numpy()}).reset_index(drop=True)
Test = pd.concat([df_base_test_X.reset_index(drop=True), y_val_df], axis=1, ignore_index=True)
Test.columns = desired_columns[1:]

features = Train.columns.to_list()[:-1]
target = 'is_goal'


model = xgb.XGBClassifier()
model.load_model(model_filename)
name = 'xgboost_q1_tune'
ut.plot_calibrations_task5(model, features, target, Val, name)

# parameters_log = {"n_estimators":400, "max_depth":5, "learning_rate":0.1,
#                   "subsample":0.7, "colsample_bytree":1,  "reg_lambda":0.65, "seed":seed}

parameters_log = grid_search.best_params_
parameters_log['seed'] = seed

metrics_log = {"accuracy": acc, "f1_score": fsc,
               "precision_score":psc, "balanced_accuracy_score":balanced_acc,
               "recall_score": rsc, "area under the curve":auc}


experiment.log_table("Train_norm.csv", Train)
experiment.log_table("Val_norm.csv", Val)
experiment.log_table("Test_norm.csv", Test)
experiment.log_table("adv_prob_q1_tune_confusion_matrix.png", cfn_mat.tolist())
experiment.log_table("probabilities_tune.csv", results)


experiment.log_parameters(parameters_log)
experiment.log_metrics(metrics_log)
experiment.log_model(best_xgb_classifier, model_filename, overwrite=True)
experiment.log_image("adv_prob_q1_tune_confusion_matrix.png", "Confusion_matrix")
experiment.log_image("dataset_balance.png", "Dataset unbalanced")
experiment.log_image("xgboost_q1_tune_diagrams.png", "Diagrams")

experiment.log_confusion_matrix(y_val, y_pred,
    title="Confusion Matrix: Evaluation",
    file_name='adv_prob_q1_tune_confusion_matrix.png')

experiment.end()
print('Experiment Ended!')
######################################
