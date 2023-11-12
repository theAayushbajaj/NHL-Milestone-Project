import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

def shot_by_distance(df_viz_game_data):
    shot_dist_figure = plt.figure(figsize=(10, 10))
    ax = sns.histplot(df_viz_game_data, x="shot_distance", hue="is_goal", multiple="stack", bins=50)
    plt.xlabel('Shot distance (ft)')
    plt.ylabel('Shot count')
    plt.title('Histogram of shot counts, binned by distance')
    ax.legend(['Goal', 'No goal'], title="Result")
    plt.savefig("/assets/images/shot_by_distance.PNG")
    
def shot_by_angle(df_viz_game_data):
    shot_angle_figure = plt.figure(figsize=(10, 10))
    ax = sns.histplot(df_viz_game_data, x="shot_angle", hue="is_goal", multiple="stack", bins=50)
    plt.xlabel('Shot angle (degree)')
    plt.ylabel('Shot count')
    plt.title('Histogram of shot counts, binned by shot angle')
    ax.legend(['Goal', 'No goal'], title="Result")
    plt.savefig("/assets/images/shot_by_angle.PNG")
    
def shot_by_distance_and_angle(df_viz_game_data):
    shot_dist_angle_figure = plt.figure(figsize=(20, 20))
    ax = sns.jointplot(data=df_viz_game_data, x="shot_distance", y="shot_angle", kind="hist")
    ax.ax_joint.set_xlabel("Shot distance (ft)")
    ax.ax_joint.set_ylabel("Shot angle (degree)")
    ax.fig.tight_layout()
    plt.savefig("/assets/images/shot_by_distance_and_angle.PNG")

def goal_rate_by_distance(df_viz_game_data):
    goal_rate_dist_df = df_viz_game_data.groupby(["shot_distance"])["is_goal"].mean().to_frame().reset_index()
    ax = sns.lineplot(data=goal_rate_dist_df , x='shot_distance', y='is_goal')
    plt.xlabel('Shot distance (ft)')
    plt.ylabel('Goal rate')
    plt.xticks(np.arange(0, 220, 20))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.title('Relation between goal rate and shot distance')
    plt.savefig("/assets/images/goal_rate_by_distance.PNG")
    
def goal_rate_by_angle(df_viz_game_data):
    goal_rate_angle_df = df_viz_game_data.groupby(["shot_angle"])["is_goal"].mean().to_frame().reset_index()
    ax = sns.lineplot(data=goal_rate_angle_df , x='shot_angle', y='is_goal')
    plt.xlabel('Shot angle (degree)')
    plt.xticks(np.arange(0, 220, 20))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.ylabel('Goal rate')
    plt.title('Relation between goal rate and shot angle')
    plt.savefig("/assets/images/goal_rate_by_angle.PNG")
    
def goal_non_empty_net_by_distance(df_viz_game_data):
    goal_entry_df = df_viz_game_data[df_viz_game_data['is_goal']==True]
    non_empty_goal_df = goal_entry_df[goal_entry_df['emptyNet']==False]
    empty_goal_df = goal_entry_df[goal_entry_df['emptyNet']==True]
    goal_figure = plt.figure(figsize=(10, 15))
    plt.subplot(211)
    ax = sns.histplot(data=non_empty_goal_df, x='shot_distance', bins=50)
    plt.xticks(np.arange(0, 220, 20))
    plt.xlabel('Shot distance (ft)')
    plt.ylabel('Goal count')
    plt.title('Histogram of goal counts by distance - Non-empty net')
    plt.subplot(212)
    ax = sns.histplot(data=empty_goal_df, x='shot_distance', bins=50)
    plt.xticks(np.arange(0, 220, 20))
    plt.xlabel('Shot distance (ft)')
    plt.ylabel('Goal count')
    plt.title('Histogram of goal counts by distance - Empty net')
    plt.savefig("/assets/images/goal_non_empty_net_by_distance.PNG")
    
    defensive_zone_df = non_empty_goal_df[non_empty_goal_df['shot distance']>100]
    desirable_columns = ['game_date','period','periodTime','game_id','is_goal','emptyNet','home_team_name','away_team_name',
                         'team_name','x_coordinate','y_coordinate','Shooter','rinkSide','shot_distance']
    
    return defensive_zone_df[desirable_columns].rename(columns={'rinkSide':'netSide'})
    
df_viz_game_data = pd.read_csv("df_data_nhl.csv")
# shot_by_distance(df_viz_game_data)
# shot_by_angle(df_viz_game_data)
# shot_by_distance_and_angle(df_viz_game_data)
# goal_rate_by_distance(df_viz_game_data)
# goal_rate_by_angle(df_viz_game_data)
wrong_df = goal_non_empty_net_by_distance(df_viz_game_data)
wrong_df.to_csv('wrong_coord.csv', index=False)
