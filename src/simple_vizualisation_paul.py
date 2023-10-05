import numpy as np
import pandas as pd

import simple_vizualisation_paul_functions as deez


#%%

dfs_dict['2016_playoff_games'].reset_index()


# initialize full dataframe
full_Data = pd.DataFrame()

for name, df in dfs_dict.items():
    season = name[:4] + "-" + str(int(name[:4]) + 1)
    df['season'] = season
    df['season type'] = name[5:]
    full_Data = pd.concat([full_Data, df], axis=0)
    
    
full_Data = full_Data.reset_index()
full_Data = full_Data.drop('index', axis = 1)
full_Data.shape




#%%

mask = (full_Data[['about_goals_away', 'about_goals_home']] == (0, 0)).all(axis=1)
tmp = full_Data[mask]




#%%
# CEHCK FOR DUPLICATES
full_Data[full_Data['result_description'].duplicated()]




#%%
data = full_Data.copy()
# 1) types of shots


data.groupby(['result_event', 'result_secondaryType']).size()

data.groupby(['result_secondaryType', 'result_event']).size().unstack()

df = data.groupby(['result_secondaryType', 'result_event']).size().unstack()

df.columns = ['Goal', 'Unscored Shot']
df['Total Shots'] = df['Goal'] + df['Unscored Shot']

df['Goal Convergance Ratio'] = df['Goal'] / df['Total Shots']
df['Goal Ratio'] = df['Goal'] / df['Goal'].sum()
df


#%%
# Pivot Table
data = full_Data.copy()

data_pt = data.pivot_table(index = ['season type','result_secondaryType'],
                            columns = ['result_event', 'season'],
                            aggfunc = len,
                            values = 'about_dateTime',
                            fill_value = 0)

data_pt


#%%

full_Data = pd.read_csv('playData.csv')

#%%

data = full_Data.copy()


data_pt = data.pivot_table(index = ['game_type','shotType'],
                            columns = ['event', 'season'],
                            aggfunc = len,
                            values = 'description',
                            fill_value = 0)

data_pt


#%%

data = full_Data.copy()

data['shot_distance'] = data.apply(lambda row: deez.euclidian_distance_goal(x_shot = row['x_coordinate'], 
                                           y_shot = row['y_coordinate'], 
                                           period = row['period'],
                                           home = (row['home_team_name'] == row['team_name'])),axis=1)

#%%

tmp = data[data['shotType'] =='Wrap-around']
