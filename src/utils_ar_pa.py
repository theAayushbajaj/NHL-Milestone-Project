import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, f1_score, \
    precision_score, recall_score, classification_report, accuracy_score, \
    roc_auc_score, balanced_accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import xgboost as xgb

def plot_calibration_curve_paul(model, features, target, val, train, name):
    # fit the model
    model.fit(train[features], train[target])
    select_features = '-'.join(features)

    joblib.dump(model, f'{name}_{select_features}.pkl')

    # get the probability of the prediction
    val['prob'] = model.predict_proba(val[features])[:, 1]

    # get the goal rate
    goal_rate = val.groupby(pd.qcut(val['prob'], 10))['is_goal'].mean()
    # take the val['is_goal']
    # find the qcut of goal probability (specifies the range for grouping)
    # group by the above metric
    # find the mean of the result

    # convert interval index to its midpoint values
    goal_rate.index = goal_rate.index.map(lambda x: x.mid)

    # get the cumulative sum of goals
    cumulative_goals = val.groupby(pd.qcut(val['prob'], 10))['is_goal'].sum().cumsum()

    # convert to proportion
    cumulative_goals = cumulative_goals / cumulative_goals.max()

    # convert interval index to its midpoint values
    cumulative_goals.index = cumulative_goals.index.map(lambda x: x.mid)

    # get the reliability diagram
    prob_true, prob_pred = calibration_curve(val[target], model.predict_proba(val[features])[:, 1], n_bins=10)
    # y = obsevered accuracy
    # x = predicted accuracy



    plt.figure(figsize=[12,9])
    # plot the goal rate
    plt.subplot(2, 2, 1)
    plt.plot(goal_rate)
    plt.xlabel('Model Probability Percentile')
    plt.ylabel('Goal Rate')

    # plot the cumulative goals
    plt.subplot(2, 2, 2)
    plt.plot(cumulative_goals)
    plt.xlabel('Model Probability Percentile')
    plt.ylabel('Cumulative Goals')

    # plot the ROC curve
    plt.subplot(2, 2, 3)
    fpr, tpr, thresholds = roc_curve(val[target], model.predict_proba(val[features])[:, 1])
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


    # plot the reliability diagram
    plt.subplot(2, 2, 4)
    plt.plot(prob_pred, prob_true)
    plt.xlabel('Model Probability')
    plt.ylabel('Actual Probability')
    plt.savefig(f"{name}_{select_features}_diagrams.png")

    plt.tight_layout()
    plt.show()
    return model


def plot_calibration_cal_paul(model, features, target, val, train, name):
    # fit the model
    model.fit(train[features], train[target])
    select_features = '-'.join(features)

    joblib.dump(model, f'{name}_{select_features}.pkl')

    # get the probability of the prediction
    val['prob'] = model.predict_proba(val[features])[:, 1]

    # get the goal rate
    goal_rate = val.groupby(pd.qcut(val['prob'], 10))['is_goal'].mean()
    # take the val['is_goal']
    # find the qcut of goal probability (specifies the range for grouping)
    # group by the above metric
    # find the mean of the result

    # convert interval index to its midpoint values
    goal_rate.index = goal_rate.index.map(lambda x: x.mid)

    # get the cumulative sum of goals
    cumulative_goals = val.groupby(pd.qcut(val['prob'], 10))['is_goal'].sum().cumsum()

    # convert to proportion
    cumulative_goals = cumulative_goals / cumulative_goals.max()

    # convert interval index to its midpoint values
    cumulative_goals.index = cumulative_goals.index.map(lambda x: x.mid)

    # get the reliability diagram
    prob_true, prob_pred = calibration_curve(val[target], model.predict_proba(val[features])[:, 1], n_bins=10)
    # y = obsevered accuracy
    # x = predicted accuracy
    fpr, tpr, thresholds = roc_curve(val[target], model.predict_proba(val[features])[:, 1])

    return goal_rate, cumulative_goals, prob_true, prob_pred, fpr, tpr, thresholds, model



def plots(goal_rate_list, cumulative_goals_list, prob_true_list,
          prob_pred_list, fpr_list, tpr_list, thresholds_list, name):

    plt.figure(figsize=[12,9])
    # plot the goal rate
    plt.subplot(2, 2, 1)
    plt.plot(goal_rate_list[0], label = 'shot_distance', color = 'blue')
    plt.plot(goal_rate_list[1], label = 'shot_angle', color = 'red')
    plt.plot(goal_rate_list[2], label = 'shot_distance and shot_angle',
             color = 'black')
    plt.plot(goal_rate_list[3], label = 'random model',
             color = 'green')
    plt.legend()
    plt.xlabel('Model Probability Percentile')
    plt.ylabel('Goal Rate')

    # plot the cumulative goals
    plt.subplot(2, 2, 2)
    plt.plot(cumulative_goals_list[0], label = 'shot_distance', color = 'blue')
    plt.plot(cumulative_goals_list[1], label = 'shot_angle', color = 'red')
    plt.plot(cumulative_goals_list[2], label = 'shot_distance and shot_angle', color = 'black')
    plt.plot(cumulative_goals_list[3], label = 'random model', color = 'green')
    plt.legend()
    plt.xlabel('Model Probability Percentile')
    plt.ylabel('Cumulative Goals')

    # plot the ROC curve
    plt.subplot(2, 2, 3)
    plt.plot(fpr_list[0], tpr_list[0], label = 'shot_distance', color = 'blue')
    plt.plot(fpr_list[1], tpr_list[1], label = 'shot_angle', color = 'red')
    plt.plot(fpr_list[2], tpr_list[2], label = 'shot_distance and shot_angle', color = 'black')
    plt.plot(fpr_list[3], tpr_list[3], label = 'random model', color = 'green')
    plt.legend()
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # plot the reliability diagram
    plt.subplot(2, 2, 4)
    plt.plot(prob_pred_list[0], prob_true_list[0], label = 'shot_distance', color = 'blue')
    plt.plot(prob_pred_list[1], prob_true_list[1], label = 'shot_angle', color = 'red')
    plt.plot(prob_pred_list[2], prob_true_list[2], label = 'shot_distance and shot_angle', color = 'black')
    plt.plot(prob_pred_list[3], prob_true_list[3], label = 'random model', color = 'green')
    plt.legend()
    plt.xlabel('Model Probability')
    plt.ylabel('Actual Probability')
    plt.savefig(f"{name}_diagrams.png")

    plt.tight_layout()
    plt.show()


def feature_selector(df_base_train, df_base_test, cols, target, random_state):
    # features
    df_base_train_X = df_base_train.loc[:, cols]
    df_base_test_X = df_base_test.loc[:, cols]
    # targets
    df_base_train_y = df_base_train.loc[:, target]
    df_base_test_y = df_base_test.loc[:, target]

    value_counts = df_base_train.is_goal.value_counts()
    sns.barplot(x=value_counts.index, y=value_counts.values)
    plt.savefig('dataset_balance.png')

    X_train, X_val, y_train, y_val = train_test_split(df_base_train_X,
                                                      df_base_train_y,
                                                      random_state=random_state,
                                                      test_size=0.2,
                                                      stratify=df_base_train_y)

    # normalize the data for angle and shot distance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(df_base_test_X)

    cols = cols + [target]
    train = pd.DataFrame(columns=cols, index=range(X_train.shape[0]))
    val = pd.DataFrame(columns=cols, index=range(X_val.shape[0]))
    test = pd.DataFrame(columns=cols, index=range(X_test.shape[0]))

    for index, feature in enumerate(cols[:-1]):
        train[feature] = X_train[:, index]
        val[feature] = X_val[:, index]
        test[feature] = X_test[:, index]

    train[target] = y_train.to_numpy()
    val[target] = y_val.to_numpy()
    test[target] = df_base_test_y.to_numpy()
    return train, val, test

def evaluate(model, features, target, val, train, name):
    # fit the model
    # model.fit(train[features], train[target])
    select_features = '-'.join(features)
    model_filename = f'{name}_{select_features}.pkl'
    loaded_model = joblib.load(model_filename)
    # get the predictions
    val['pred'] = loaded_model.predict(val[features])

    # get the probability of the prediction
    val[['prob_not_goal', 'prob_goal']] = loaded_model.predict_proba(val[features])
    y_prob = val[['prob_not_goal', 'prob_goal']]
    y_val = val['is_goal']
    y_pred = val['pred']

    results = pd.DataFrame({'is_goal': y_val,
                            'is_goal_pred': y_pred,
                            'prop_goal': y_prob.iloc[:, 1],
                            'prob_not_goal': y_prob.iloc[:, 0]})

    select_features = '-'.join(features)

    results.to_csv(f'{name}{select_features}.csv')
    cfn_mat = confusion_matrix(y_val, y_pred)

    labels = ['not_goal', 'is_goal']
    sns.heatmap(cfn_mat, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(f"{name}{select_features}_confusion_matrix.png")
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

    metrics_log = {"accuracy": acc, "f1_score": fsc,
                   "precision_score": psc, "balanced_accuracy_score": precision_score,
                   "recall_score": rsc, "area under the curve": auc,
                   'report': report}

    return cfn_mat, y_val, y_pred, results, metrics_log


def renamer(df):
    df = df.copy()
    cols = df.columns.to_list()
    cols_space = [col.replace(' ', '_') for col in cols]
    cols_rename = {cols[i]: cols_space[i] for i in range(len(cols))}
    return cols_rename


class RandomModel_paul:
    def __init__(self):
        pass
    def predict_proba(self, X):
        col_1 = np.random.uniform(size=len(X))
        col_2 = 1 - col_1
        return np.column_stack((col_1, col_2))
    def fit(self, X, y):
        pass

# def RebuildDataFrame(X, y, columns):
#     """This function recreates a dataframe from X_train and y_train created from
#     train_test_split"""
#     Data = pd.DataFrame(columns=columns[1:], index=range(X.shape[0] + 1))
#     for i, col in enumerate(columns[1:-1]):
#         Data.loc[:, col] = X[:, i]
#     Data.loc[:, col] = y.to_numpy()
#     return Data


def plot_calibrations_task5(model, features, target, val, name):
    # fit the model

    # get the probability of the prediction
    val['prob'] = model.predict_proba(val[features])[:, 1]

    # get the goal rate
    goal_rate = val.groupby(pd.qcut(val['prob'], 10))['is_goal'].mean()
    # take the val['is_goal']
    # find the qcut of goal probability (specifies the range for grouping)
    # group by the above metric
    # find the mean of the result

    # convert interval index to its midpoint values
    goal_rate.index = goal_rate.index.map(lambda x: x.mid)

    # get the cumulative sum of goals
    cumulative_goals = val.groupby(pd.qcut(val['prob'], 10))['is_goal'].sum().cumsum()

    # convert to proportion
    cumulative_goals = cumulative_goals / cumulative_goals.max()

    # convert interval index to its midpoint values
    cumulative_goals.index = cumulative_goals.index.map(lambda x: x.mid)

    # get the reliability diagram
    prob_true, prob_pred = calibration_curve(val[target], model.predict_proba(val[features])[:, 1], n_bins=10)
    # y = obsevered accuracy
    # x = predicted accuracy
    fpr, tpr, thresholds = roc_curve(val[target], model.predict_proba(val[features])[:, 1])

    plt.figure(figsize=[12,9])
    # plot the goal rate
    plt.subplot(2, 2, 1)
    plt.plot(goal_rate)
    plt.xlabel('Model Probability Percentile')
    plt.ylabel('Goal Rate')

    # plot the cumulative goals
    plt.subplot(2, 2, 2)
    plt.plot(cumulative_goals)
    plt.xlabel('Model Probability Percentile')
    plt.ylabel('Cumulative Goals')

    # plot the ROC curve
    plt.subplot(2, 2, 3)
    fpr, tpr, thresholds = roc_curve(val[target], model.predict_proba(val[features])[:, 1])
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    # plot the reliability diagram
    plt.subplot(2, 2, 4)
    plt.plot(prob_pred, prob_true)
    plt.xlabel('Model Probability')
    plt.ylabel('Actual Probability')
    plt.savefig(f"{name}_diagrams.png")

    plt.tight_layout()
    plt.show()

