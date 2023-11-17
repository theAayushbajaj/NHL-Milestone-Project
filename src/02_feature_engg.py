import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class Penalty:
    def __init__(self, start_time_in_secs, penalty_time_in_minutes, home_team):
        self.start_time_in_secs = start_time_in_secs
        self.end_time_in_secs = start_time_in_secs + penalty_time_in_minutes*60
        self.end_first_minor = start_time_in_secs + 2*60
        self.major = penalty_time_in_minutes==5
        self.double_minor = penalty_time_in_minutes==4
        self.home_team = home_team
        self.avail = True
        
    def alter_game_information(self, curr_time_in_secs, is_goal):
        # If the penalty duration elapses, change the state to unavailable
        if(curr_time_in_secs >= self.end_time_in_secs):
            self.avail = False
            return
        # If the penalty duration hasn't passed, verify 'is_goal' value
        if(is_goal):
            # If it's a significant penalty, the penalty will persist
            if(self.major): return
            elif(self.double_minor):
                # The initial minor penalty time hasn't passed yet
                if(curr_time_in_secs < self.end_first_minor): self.end_time_in_secs = curr_time_in_secs + 2 * 60
                # The initial minor penalty duration has already concluded, thus the second one will end instantly
                else: self.avail = False
            # This is a minor penalty, and hence it concludes after a single shot
            else: self.avail = False
        

def read_data_from_json_file(file_path) -> dict:
    """
    A utility function that accesses the contents of a JSON file for a specific game using the given file path
    :param file_path: File pathway to access the game data
    :return: A directionary holding the data information from the JSON file
    """
    if(os.path.exists(file_path)):
        with open(file_path) as f: return json.load(f)
    else:
        print("Unable to find the game data in: " + file_path)
        return
    
    
def game_data_np(file_path, features_set):
    """
    Retrieve all shot and goal occurrences in a specific game and transform them into a numpy array
    :param file_path: game data file path
    :param features_set: collection of intended features
    :return: A numpy array containing data regarding all shot and goal occurrences
    """
    all_game_data, game_feats = read_data_from_json_file(file_path), np.array(features_set)    
    all_plays = all_game_data['liveData']['plays']['allPlays']
    # If the file contains no data, return a list that is empty
    if(len(all_plays)==0): return []
    try: first_period_home_side = all_game_data['liveData']['linescore']['periods'][0]['home']['rinkSide']
    except: return []
    
    game_played_date, game_played_id = all_game_data['gameData']['datetime']['dateTime'][0:10], all_game_data['gamePk']
    n_season = str(game_played_id)[0:4]
    away_team, home_team = all_game_data['gameData']['teams']['away']['triCode'], all_game_data['gameData']['teams']['home']['triCode']
    # The location of the goals scored by both teams
    left_goal_post_coords, right_goal_post_coords = np.array([-89,0]), np.array([89,0])
    # Set up the coordinates (if incorrect, will be corrected during feature engineering).
    home_goal_post_coords = right_goal_post_coords
    away_goal_post_coords = left_goal_post_coords
    penalties, penalty_start_time_in_secs, curr_penalty_stat, power_play_time = [], 0, False, 0
    
    # Iterate over all the events
    for event_index, event in enumerate(all_plays):
        event_type = event['result']['event']
        is_goal = event_type == 'Goal'
        goal_home = event['about']['goals']['home']
        goal_away = event['about']['goals']['away']
        # Current event's game time information
        time_period, period = event['about']['periodTime'], event['about']['period']
        time_period_in_mins, time_period_in_secs = int(time_period.split(':')[0]), int(time_period.split(':')[1])
        game_time_in_secs = (period-1)*20*60 + time_period_in_mins*60 + time_period_in_secs
        # Steps to manage the penalty event
        if(event_type == "Penalty"):
            if(len(penalties)==0):
                penalty_start_time_in_secs, curr_penalty_stat = game_time_in_secs, True
            penalty_minutes = event['result']['penaltyMinutes']
            is_team_home = event['team']['triCode'] == home_team
            penalties.append(Penalty(game_time_in_secs, penalty_minutes, is_team_home))
        n_home_team, n_away_team = 5, 5
        for penalty_instance in penalties:
            if(penalty_instance.avail):
                if(penalty_instance.home_team): n_home_team -= 1
                else: n_away_team -= 1
            else: penalties.remove(penalty_instance)
            penalty_instance.alter_game_information(game_time_in_secs, is_goal)
        if(curr_penalty_stat):
            power_play_time = game_time_in_secs - penalty_start_time_in_secs
        time_power_play_store = power_play_time
        
        # If there's an ongoing penalty and the number of players in both teams returns to 5, then the penalty concludes
        if(curr_penalty_stat and n_home_team==5 and n_away_team==5):
            curr_penalty_stat, power_play_time = False, 0
        
        # Previous event's game time information
        period_time_last_event = all_plays[event_index-1]['about']['periodTime']
        period_last_event = all_plays[event_index-1]['about']['period']
        period_minutes_last_event = int(period_time_last_event.split(':')[0])
        period_seconds_last_event = int(period_time_last_event.split(':')[1])
        game_seconds_last_event = (period_last_event-1)*20*60 + period_minutes_last_event*60 + period_seconds_last_event
        # Time difference between the current event and the last event
        time_from_last_event = game_time_in_secs - game_seconds_last_event
        # Change sides
        if((all_plays[event_index-1]['about']['period'] != period and period!=1)):
            goal_home_coordinates_current = home_goal_post_coords
            home_goal_post_coords = away_goal_post_coords
            away_goal_post_coords = goal_home_coordinates_current
        # Only filter for Goals or Shots
        if(event_type not in ['Goal', 'Shot']):
            continue
        team_shot = event['team']['triCode']
        if(is_goal):
            is_empty_net = event['result']['emptyNet'] if 'emptyNet' in event['result'] else False
            strength = event['result']['strength']['name'] if 'strength' in event['result'] else ""
        else:
            is_empty_net, strength = False, ""
        # To determine if the shot or goal is rebound
        is_rebound = False
        # A shot or goal qualifies as a rebound if it originates from a blocked shot by the same team
        if((all_plays[event_index-1]['result']['event'] in ["Blocked Shot","Shot"])
           and (team_shot==all_plays[event_index-1]['team']['triCode']) and time_from_last_event<5):
            is_rebound = True
        # To determine if there is any coordinates information missing
        is_corr_available = all(cor in event['coordinates'] for cor in ['x', 'y'])
        coordinates = [event['coordinates']['x'], event['coordinates']['y']] if is_corr_available else ""
        # Do not consider events with missing or no coordinates
        if(len(coordinates)==0): continue
        shot_type = event['result']['secondaryType'] if 'secondaryType' in event['result'] else ""
        goalie = ""
        for player in event['players']:
            if(player['playerType'] == "Goalie"):
                goalie = player['player']['fullName']
                continue
            if(player['playerType'] in ["Scorer", "Shooter"]):
                shooter = player['player']['fullName']
        type = 'home' if home_team == team_shot else 'away'
        if len(all_game_data['liveData']['linescore']['periods']) > 0 and period <= 4:
            rink_side = all_game_data['liveData']['linescore']['periods'][period-1][type]['rinkSide']
        x_coords, y_coords = coordinates[0], coordinates[1]     
        
        # Incorporates previous event information
        last_event_type = all_plays[event_index-1]['result']['event']
        is_corr_available_last_event = all(cor in all_plays[event_index-1]['coordinates'] for cor in ['x', 'y'])
        last_event_coordinates = [all_plays[event_index-1]['coordinates']['x'], all_plays[event_index-1]['coordinates']['y']] if is_corr_available_last_event else ""
        if(last_event_coordinates!=''):
            x_last_event = last_event_coordinates[0]
            y_last_event = last_event_coordinates[1]
        else:
            x_last_event = ""
            y_last_event = ""
        
        # a particular event
        event_data = [n_season, game_played_date, period, time_period, game_played_id, home_team, away_team,
                      is_goal, team_shot, x_coords, y_coords, shooter, goalie, shot_type,
                      is_empty_net, strength, goal_home, goal_away, is_rebound, rink_side,
                      game_time_in_secs, last_event_type, x_last_event, y_last_event, time_from_last_event,
                      n_home_team, n_away_team, time_power_play_store]
        game_feats = np.vstack((game_feats, event_data))
    # The first row of game_feats is just the header
    game_events_data_df = pd.DataFrame(data=game_feats[1:], columns=game_feats[0])
    game_feats = game_events_data_df.values

    return game_feats


def obtain_files_list(dir_path):
    """
    A helper function to retrieve the collection of all files within a directory and its sub-folders
    :param dir_path: directory path
    :return: A collection with all files within the directory and its sub-directories
    """
    # Retrieve all the files and folders within the specified main directory
    collate_entries_main_dir, collate_files = os.listdir(dir_path), list()    
    # Iterate over all the sub-folders
    for dir_part in collate_entries_main_dir:
        dir_part_path = os.path.join(dir_path, dir_part)
        # Determine if 'dir_part' is a file or a directory
        if(os.path.isdir(dir_part_path)): collate_files = collate_files + obtain_files_list(dir_part_path)
        else: collate_files.append(dir_part_path)
    return collate_files

def collate_game_info(dir_path, features_set):
    """
    Retrieve all shots and goals from all games within a directory and its subfolders
    :param dir_path: Game data directory path
    :param features_set: collection of all intended features
    :return: dataframe encompassing details of all shots and goals found in the directory
    """
    collate_files, all_games_data = obtain_files_list(dir_path), np.array(features_set)
    for path_to_file in tqdm(collate_files):
        retrieve_game_data = game_data_np(path_to_file, features_set)
        if(len(retrieve_game_data)==0): continue
        all_games_data = np.vstack((all_games_data, retrieve_game_data))        
    df_game_events = pd.DataFrame(data=all_games_data[1:], columns=all_games_data[0])
    return df_game_events

def prepare_tidy_data(game_seasons_list, raw_dir_path):
    """
    A supporting function to prepare tidy data with information of last events
    :param game_seasons_list: collection containing seasons
    :param raw_dir_path: path to the raw data directory
    :return: dataframe containing tidy data with information of last events
    """
    features_set = ['season','game_date','period','periodTime','game_id','home_team_name','away_team_name',
                    'is_goal','team_name','x_coordinate', 'y_coordinate','Shooter','Goalie','shotType','emptyNet',
                    'strength','home_goal','away_goal','is_rebound', 'rinkSide','game_seconds','last_event_type', 
                    'x_last_event', 'y_last_event', 'time_from_last_event','num_player_home', 'num_player_away', 'time_power_play']
    game_tidy_df = pd.DataFrame(columns=features_set)
    for season in game_seasons_list:
        print(f"Retrieving season {season} data")
        dir_path = os.path.join(raw_dir_path, season)
        print(f"Printing path to the directory: {dir_path}")
        collate_game_df = collate_game_info(dir_path, features_set)
        game_tidy_df = pd.concat([game_tidy_df, collate_game_df])
    return game_tidy_df
    

current_dir_path = os.getcwd()
print(current_dir_path)
raw_dir_path = os.path.join(os.getcwd(), "data", "raw_data", "playoffs")
print(f'raw_dir_path: {raw_dir_path}')
# game_seasons_list = ['2016', '2017', '2018', '2019', '2020']
game_seasons_list = ['2020']
df_tidy_data = prepare_tidy_data(game_seasons_list, raw_dir_path)
df_tidy_data.to_csv('playoff_2020.csv',index=False)
