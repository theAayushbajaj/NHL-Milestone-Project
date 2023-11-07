import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

def separate(season_threshold, df):
    df = df.copy()
    df_train = df[df['season'] < season_threshold] # less than 2020 for train
    df_test = df[df['season'] == season_threshold] # 2020 for test
    return df_train.reset_index(), df_test.reset_index()

def distribution(y, name):
    counts = y.value_counts()
    goals = counts.loc[1] / (counts.sum())
    non_goals = counts.loc[0] / (counts.sum())
    plt.figure(figsize=[8, 6])
    plt.bar(x=['is goal', 'not a goal'], height=[goals, non_goals])
    plt.title(f'goal distribution percentage in 2016-2019 for {name}')
    plt.ylabel('count')
    plt.ylim((0, 1))
    plt.savefig('goal distribution percentage in 2016-2019 for {}.png'.format(name),
                dpi=200)
    plt.show()
    pass

def heatmap(cnf_matrix):
    plt.figure(figsize=[8, 6])
    ticks = [[0, 1], ['not a goal', 'is a goal']]
    plt.xticks(ticks[0], ticks[1])
    plt.yticks(ticks[0], ticks[1])
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g')
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.text(0.5, 257.44, 'Predicted label')
    plt.savefig('Baseline confusion matrix.png', dpi=200)
    plt.show()

if __name__=='main':
    pass