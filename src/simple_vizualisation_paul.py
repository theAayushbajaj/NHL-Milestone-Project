import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import src.simple_vizualisation_paul_functions as deez


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

full_Data = pd.read_csv('src/data/raw_data/playData.csv')

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

data

#%%

index_features = ['shotType']
column_features = ['event']
agg_func_list = ['quantile']


data_pt = data.pivot_table(index = index_features,
                            columns = column_features,
                            aggfunc = agg_func_list,
                            values = 'shot_distance',
                            fill_value = 0)

data_pt



#%%

df = data_pt
# Reshape the DataFrame for easier plotting
df_melted = df['mean'].reset_index()

sns.barplot(x="Shot", y="shotType", hue="game_type",  data=df_melted)


df_melted.plot(kind = 'bar')


#%%

data

sns.barplot(x = 'shot_distance', y = 'shotType', hue = 'event' ,data = data)
sns.catplot(x = 'shot_distance', y = 'shotType', hue = 'event' ,data = data,
            kind = 'box')



#%%

# QUNATILE SEPARATION

data['shot_distance_quantile'] = pd.qcut(data['shot_distance'] , 10)

#%%

# sns.barplot(x = 'shot_distance_quantile', y = 'shotType', hue = 'event' ,data = data)

index_features = ['shot_distance_quantile']
column_features = ['season','event']
agg_func_list = [len]


data_pt = data.pivot_table(index = index_features,
                            columns = column_features,
                            aggfunc = agg_func_list,
                            values = 'eventIdx',
                            fill_value = 0)

data_pt.columns = data_pt.columns.droplevel(0)

data_pt = data_pt.stack(level = 0)
#%%

data_pt['conversion_rate'] = data_pt['Goal']/ (data_pt['Goal'] + data_pt['Shot'])

data_pt

data_pt = data_pt.unstack()
#%%
data_pt = data_pt.reset_index()

#%%

# Calculate the midpoints of the intervals
midpoints = [round((interval.left + interval.right) / 2, 2) for interval in data_pt.shot_distance_quantile]

plt.figure(figsize=(12, 6))
sns.barplot(x=midpoints, y='conversion_rate' , hue = 'season', data = data_pt)
plt.xlabel('Shot Distance Midpoint')
plt.ylabel('Conversion Rate')
plt.title('Conversion Rate by Shot Distance Midpoint')  # adjust x-tick labels
plt.show()




#%%

#3)

data = full_Data.copy()

data['shot_distance'] = data.apply(lambda row: deez.euclidian_distance_goal(x_shot = row['x_coordinate'], 
                                           y_shot = row['y_coordinate'], 
                                           period = row['period'],
                                           home = (row['home_team_name'] == row['team_name'])),axis=1)

data

#%%
# Divide in equally seperated boxes
data['shot_distance_section'] = pd.qcut(data['shot_distance'] , 8)

#%%

features = ['event', 'shotType', 'shot_distance', 'shot_distance_section']

df = data[features]

tmp = df.groupby(['shotType', 'shot_distance_section']).apply(deez.get_stats_q3)


# Now, apply the function
tmp = tmp.groupby(['shot_distance_section'], as_index = False).apply(calculate_proportion, column_name='total_shots')
tmp = tmp.groupby(['shot_distance_section'], as_index = False).apply(calculate_proportion, column_name='total_goal')

tmp = tmp.fillna(0)

tmp = tmp.droplevel([0, 1])

tmp = tmp.reset_index()
tmp.columns
'''Index(['shotType', 'shot_distance_section', 'total_shots', 'total_goal',
       'conversion_rate', 'total_shots_prop', 'total_goal_prop'],
      dtype='object')'''
    
tmp['conversion_rate'][tmp['total_shots'] < 50] = 0   



#%%


# Calculate the midpoints of the intervals
midpoints = [round((interval.left + interval.right) / 2, 2) for interval in tmp.shot_distance_section]

ax = sns.barplot(x=midpoints, y='conversion_rate', hue='shotType', data=tmp)
# Set y ticks from 0 to 0.4
ax.set_ylim(0, 0.4)

# Show the plot
plt.show()












