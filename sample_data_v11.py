# import libraries
from ast import Tuple
import json
from xmlrpc.client import boolean
import numpy as np
import pandas as pd
import requests
import os
import math
import sklearn
from sklearn.preprocessing import MinMaxScaler
# from preprocessed_modified import downloadData
import preprocess_modified

def save_json(game_id:str, data: dict, path:str)->None:
    """
    Saves the json file with the same name as the game_id in the given path.

    Parameters: 
    game_id (str): the game_id of the desired json file to be saved. 
    data (dict): the input json file
    path: 

    Returns: 
    None 
    """
    json_object = json.dumps(data, indent=4)
    file_direction = os.path.join(path, f'{game_id}.json')
    with open(file_direction, 'w') as file:
        file.write(json_object)
    pass


def open_json(game_id: str, path:str)->dict:
    """
    Opens the json file with a given game_id and a path. 

    Parameters: 
    game_id: a game id which consisted of 10 digits 
    path: the path where the json file is sotored in. 

    Returns: 
    json_data: returns the json file. 
    """
    file_path = os.path.join(path, fr'{game_id}.json')
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data


def change_directory(main_folder_name=r'NHL_data')->str:
    """
    Changes the directory from anywhere to the `NHL_data` folder. 

    Parameters: 
    main_folder_name (str): default is `NHL_data`. 

    Returns: 
    path (str): the desired path of the given game_id to store its files
    """
    current = os.getcwd()
    folder = os.path.join(current, main_folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        pass
    return folder


def save_csv(game_id, df: pd.DataFrame, name):
    """
    Saves the given dataframe with name consisted of the game_id and a desired name in the input. 

    Parameters: 
    game_id (str)
    df (pd.DataFrame): the dataframe to be saved 
    name (str): the name of the dataframe to be saved which is `f{game_id}-{name}.csv`. 

    Returns: 
    None 
    """
    folder = change_directory()
    path_csv = os.path.join(folder, f'{game_id}-{name}.csv')
    df.to_csv(path_csv)

# functions
def download_single_data(game_id:str)-> dict:
    """
    Downloads a single game data with the given game_id. 

    Parameters: 
    game_id (str): desired game with the given game_id 

    Returns: 
    json_data: if the game with the given game_id existed, a dictionary containing the game data with the given game_id
    """
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    try:
        response = requests.get(url)
        # raise an error if the response was unsuccessful
        response.raise_for_status()
        json_data = response.json()
        return json_data
    except Exception as e:
        print(f'The {e} occurred!')
        # return None

def tidy_data(game_id: str, data: dict) -> pd.DataFrame:
    """
    Conversts Json file to a pandas dataframe 

    Parameters: 
    game_id: a string object of 10 digits 
    data: a json file involving raw data collected from NHL ALI 

    Returns: 
    df: a pandas dataframe contraing desired features; 
    'total_period': number of periods played in the game (int)
    'season': the season at which the game was played (str)
    'game_id': the unique game identifier (str)
    'goal': if the shot was a goal or not (bool)
    'period': the period at which the event occurred (int)
    'periodType': the typr of the period at which the event occurred (str)
    'x_coordinate': the x coordinate of the shot (int)
    'y_coordinate': the y coordinate of the shot (int)
    'eventOwnerTeamId': the id of the team who made the shot (str)
    'situationCode': a string object consisted of 4 digits summarizing the presence of goalie and players of away and home team 
    'away_team_id': the unique id of the away team (str)
    'home_team_id': the unique id of the home team (str)
    'away_team_name': the name of the away team (str)
    'home_team_name': the name of the home team (str)
    'zoneCode': the zone of event from the event owner's perspective (str)
    'homeTeamDefendingSide': the defending side of the home team 
    """
    desired_game = game_id
    away_team_id = data['awayTeam']['id']
    away_team_name = data['awayTeam']['name']['default']

    home_team_id = data['homeTeam']['id']
    home_team_name = data['homeTeam']['name']['default']
    num_period = data['period']

    Plays = [element for element in data['plays'] if element['typeDescKey'] in ['goal', 'shot-on-goal']]

    goal_list = []  # 1 if goal else 0
    period_list = []
    periodType_list = []
    x_coordiante_list = []
    y_coordiante_list = []
    situationCode_list = []
    eventOwnerTeamId_list = []
    homeTeamDefendingSide_list = []
    eventOwnerTeamId_list = []
    zoneCode_list = []

    for element in Plays:
        # shot-on-goal or goal
        goal_list.append(element.get('typeDescKey', ''))
        # period number
        period_list.append(element.get('period', ''))
        # periodType
        periodDescriptor = element.get('periodDescriptor', '')
        periodType_list.append(periodDescriptor.get('periodType', ''))

        # details
        details = element.get('details', '')
        x_coordiante_list.append(details.get('xCoord', ''))
        y_coordiante_list.append(details.get('yCoord', ''))
        eventOwnerTeamId_list.append(details.get('eventOwnerTeamId', ''))

        # situationCode
        situationCode_list.append(element.get('situationCode', ''))

        # homeTeamDefendingSide
        homeTeamDefendingSide_list.append(element.get('homeTeamDefendingSide', ''))
        zoneCode_list.append(details.get('zoneCode', ''))

    length = len(Plays)
    df = pd.DataFrame({
        'total_period': [num_period] * length,
        'season': [desired_game[:4]] * length,
        'game_id': [desired_game] * length,
        'goal': goal_list,
        'period': period_list,
        'periodType': periodType_list,
        'x_coordinate': x_coordiante_list,
        'y_coordinate': y_coordiante_list,
        'eventOwnerTeamId': eventOwnerTeamId_list,
        'situationCode': situationCode_list,
        'away_team_id': [away_team_id] * length,
        'home_team_id': [home_team_id] * length,
        'away_team_name': [away_team_name] * length,
        'home_team_name': [home_team_name] * length,
        'zoneCode': zoneCode_list,
        'homeTeamDefendingSide': homeTeamDefendingSide_list
    })
    return df

def stength_extractor(row: pd.Series)->str:  # 0541: away (5); home(5) - correct logic
    """
    Identifies the strength of the event from the event owner's perspective 

    Parameters: 
    row: the row of the dataframe

    Returns: 
    code: defines the strength of the event w.r.t. to the team who is shooting
    """
    situation_code = str(row['situationCode'])
    away_sum = int(situation_code[0]) + int(situation_code[1])
    home_sum = int(situation_code[2]) + int(situation_code[3])
    eventOwnerId = row['eventOwnerTeamId']
    homeId = row['home_team_id']
    awayId = row['away_team_id']
    if eventOwnerId == homeId:
        is_home = True  # home is attacking
    else:
        is_home = False  # away is attacking

    if away_sum == home_sum:  # if the total number of players is equal
        code = 'Even'
    if away_sum > home_sum:
        if is_home:  # if home is attacking; the strenght w.r.t. home is considered
            code = 'ShortHanded'  # home is shorthanded
        else:  # if away is attacking; then strength w.r.t. away is considered
            code = 'PowerPlay'  # away is on PowerPlay
    if away_sum < home_sum:
        if is_home:  # if home is attacking; the strenght w.r.t. home is considered
            code = 'PowerPlay'  # home is on PowerPlay
        else:  # if away is attacking; then strength w.r.t. away is considered
            code = 'ShortHanded'  # away is shorthanded
    return code

def empty_net_extractor(row:pd.Series)->(bool, bool, str, str): 
    """
    Identifies whether the net to which the shot was heading. 

    Parameters: 
    row: the row of the dataframe that apply method is applied on. 

    Returns: 
    empty_net: whether the net is empty or full (string)
    is_home: whether the home team is making the shot or not (string)
    away_net: the net status of the away team (string)
    home_net: the net status of the home team (string)
    """
    eventOwnerId = row['eventOwnerTeamId']
    homeId = row['home_team_id']
    awayId = row['away_team_id']
    if eventOwnerId == homeId:
        is_home = True  # home is attacking
    else:
        is_home = False  # away is attacking
    situationCode = row['situationCode']
    goalie_home = int(str(situationCode)[-1])  # checks the existence of home team goalie
    goalie_away = int(str(situationCode)[0])  # checks the existence of away team goalie
    if goalie_home == 0:
        home_net = 'Empty'
    else:
        home_net = 'Full'
    if goalie_away == 0:
        away_net = 'Empty'
    else:
        away_net = 'Full'

    if is_home:  # home is shooting
        if home_net == 'Empty':
            empty_net = False
        elif home_net == 'Full':
            empty_net = False
        if away_net == 'Empty':  # the net of the attacked team (away) is empty
            empty_net = True
        elif away_net == 'Full':
            empty_net = False
    else:  # away is shooting
        if away_net == 'Empty':
            empty_net = False
        elif away_net == 'Full':
            empty_net = False
        if home_net == 'Empty':  # the net of the attacked team (home) is empty
            empty_net = True
        elif home_net == 'Full':
            empty_net = False
    return empty_net, is_home, away_net, home_net

def team_name_finder(row: dict) -> str:
    """
    Finds the name of the team who owns the event 

    Parameters: 
    row: the row of a dataframe which the apply method is applied on
    
    Returns: 
    shooter_team_name: the name of the team who is shooting
    """
    home_team_name = row['home_team_name']
    home_team_id = row['home_team_id']
    away_team_id = row['away_team_id']
    away_team_name = row['away_team_name']
    eventOwnerTeamId = row['eventOwnerTeamId']
    if home_team_id == eventOwnerTeamId:
        shooter_team_name = home_team_name  # home is shooting
    else: 
        shooter_team_name = away_team_name  # away is shooting
    # else:
    #     shooter_team_name = None
    return shooter_team_name


def rink_side_finder(row: pd.Series)-> str:
    """
    Finds the rink side to which the puck is heading. 
    logic: rink_side is the opposite side of the attacking team's defending side.
    example: in general (not considering rare cases where a team is shooting to their own net)
    1 - if the home team is shooting, the rink_side = opposite of homeTeamDefendingSide = home_team_attacking_side
    2 - if the away team is shooting, the rink_side = homeTeamDefendingSide

    Parameters: 
    row: the row of a dataframe that this function is being applied on. 

    Returns: 
    rink_side: The side at which the team who is shooting is aiming for -> call this 'rink_side'
    """
    home_team_id = row['home_team_id']
    away_team_id = row['away_team_id']
    home_team_defending_side = row['homeTeamDefendingSide']
    shooter_id = row['eventOwnerTeamId']
    if home_team_defending_side == 'right':
        home_team_attacking_side = 'left'
    if home_team_defending_side == 'left':
        home_team_attacking_side = 'right'
    # else:
    #     # this is for handeling the cases where the 'homeTeamDefendingSide' is None 
    #     home_team_attacking_side = None
    #     return None
    if shooter_id == home_team_id:
        # shooter_team = 'home'
        rink_side = home_team_attacking_side  # the rink side of the home offensive zone (=away defensive zone)
    # if shooter_id == away_team_id:
    else: 
        # shooter_team = 'away'
        rink_side = home_team_defending_side  # the rink side of the home defensive zone
    return rink_side


def shot_distance_calculator(row: pd.Series)->float:
    """
    Calculates the shot distance of a given shot with `x_cooridnate`, `y_coordinate`, and `rink_side`. 
    logic: x_net is either (-89) or (89).
    x_net = -89 if the rink_side of the attacking team is on the left.
    x_net = 89 if the rink_side of the attacking team is on the right.

    Paremeters: 
    row: the row of dataframe

    Returns: 
    distance: the distance which is the distance from which the shot was shooted from the net
    """
    x = row['x_coordinate']
    y = row['y_coordinate']
    rink_side = row['rink_side']
    if ((rink_side == None) or x == '' or pd.isnull(x) or y == '' or pd.isnull(y)):
        return ""
    else:
        y_net = 0
        if rink_side == 'left':
            x_net = -89
        if rink_side == 'right':
            x_net = 89
        distance = math.sqrt((x - x_net) ** 2 + (y - y_net) ** 2)
        return  round(distance)


def shot_angle_calculator(row):
    """
    Computes the shot angle of the given shot with `x_cooridnate`, `y_coordinate`, and `rink_side`. 
    logic:
    A - if the shot is made in front of the rink (not perpendecular):
    A1 - if the rink_side is on the left: theta =  arctan(|y| / (-89 - |x|))
    A2 - if the rink_side is on the right: theta =  arctan(|y| / (89 - |x|))
    B - if the shot is made behind the net: theta = (not perpendecular): theta = pi/2 + theta
    C - if the shot is perpendecular to the net (x_net = x): theta = pi/2
    D - if the shot is parallel to the vertical netline (y=0): theta = 0
    
    Parameters: 
    row: the row of the play dataframe

    Returns: 
    angle or computed_angle: the angle from which the shot was shooted from the net
    angle if the shot right was shooted from a straing centering line passing through the net, i.e., `y_coordinate` = 0 
    computed_angle otherwise. 

    """
    x = row['x_coordinate']
    y = row['y_coordinate']
    rink_side = row['rink_side']
    if ((rink_side == None) or x == '' or pd.isnull(x) or y == '' or pd.isnull(y)):
        return ""
    else:
        if rink_side == "right":
            x_net = 89
        else:
            x_net = -89
        x_dist_abs, y_dist_abs = np.abs(x_net - x), np.abs(y)
        shot_taken_behind_net = (x_net == 89 and x > 89) or (x_net == -89 and x < -89)
        shot_taken_perpendicular_net = (x_net == x)

        if (y == 0):
            angle = 0
            return angle
        else:
            if (shot_taken_perpendicular_net):
                angle = np.pi / 2
            else:
                angle = np.arctan(y_dist_abs / x_dist_abs)
                if (shot_taken_behind_net):
                    angle += np.pi / 2
            computed_angle = round(np.rad2deg(angle))
            return computed_angle

def feature_extractor_part_1(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    Extracts column features 'is_goal', 'home_strength', and 'empty_net'. 
    
    Parameters: 
    df: tided dataframe

    Returns: 
    df: Extract features (1) shot_distance, (2) shot_angle
    """
    df2 = df.copy()
    df2['is_goal'] = (df2['goal'] == 'goal') * 1
    df2['home_strength'] = df2.apply(stength_extractor, axis=1)
    df2['empty_net'], df2['is_home'], df2['away_net'], df2['home_net'] = zip(*df2.apply(empty_net_extractor, axis=1))
    return df2

def flip(side:str) -> str:
    """
    Flips the input side 

    Parameters:  
    side: side which is either only left or right 

    Returns: 
    flipped: right if side is left, and left if side is right. 
    """
    if side == 'right':
        flipped = 'left'
    elif side == 'left':
        flipped = 'right'
    else:
        raise ValueError('The side must be either right or left')
    return flipped

def feature_extractor_part_2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuilds the 'homeTeamDefendingSide' attribute 
    logic: zoneCode -> the side of the game where event is happening from the team's owner event. 
    every period the side of teams is swapped. If the side of the home team is known, it's side in other periods can simply be inferred. 
    step 1- pick the first period 
    step 2- filter for either defensive zone or offensive zone of the home team events in the first period 
    step 3- use zoneCode and x_coordinate sign to set the homeTeamDefendingSide: 
    a - if the home team was in the defensive zone, put the homeTeamDefendingSide compatible to the x_coordinate 
    example: x_coorindate = 65 -> homeTeamDefendingSide = right 
    b - if the home team was in the offensive zone, put the homeTeamDefendingSide as opposed to the x_coordinate 
    example: x_coordinate = 65 -> homeTeamDefendingSide = left 
    step 4: filter based on period, set the homeTeamDefendingSide and flipped and save it for the next period. 

    Parameters:
    df: tided dataframe

    Returns 
    df2: rebuilds the homeTeamDefendingSide feature which is not available for years below 2019 in the updated API 
    """
    df2 = df.copy()
    num_period = df['total_period'][0]  # pick one of them
    period_crn = (df2['period'] == 1) # pick the first period 
    zone_crn_D = (df2['zoneCode'] == 'D') # if the zoneCode is Defensive 
    zone_crn_O = (df2['zoneCode'] == 'O') # if the zoneCode is Offensive 
    home_team_crn = (df2['is_home'] == True) # if the eventOwnerTeam is home team 
    # x_coor_crn = (df2['x_coordinate'] != 0) # if the x_coordinate is not equal to zero 
    #TODO If we are in either in the defensive or offensive zone, the x_coordinate is never equal to zero  
    criterion_D = (period_crn & zone_crn_D & home_team_crn)
    criterion_O = (period_crn & zone_crn_O & home_team_crn)
    if criterion_D.any(): 
        # if the hometeam owns some events in their defensive zone without the x_coorindate being zero in the first period
        df_and = df2.loc[criterion_D, :].reset_index()
        cr_D = True
    if criterion_O.any():
        # if the hometeam owns some events in their offensive zone without the x_coordinate being zero in the second period 
        df_and = df2.loc[criterion_O, :].reset_index()
        cr_D = False
    x_coor = df_and.loc[0, 'x_coordinate'] # pick one of the x_coordinates 
    if cr_D:  # if home is in defensive zone
        if x_coor > 0: 
            # home was defending themselves in the right side -> homeTeamDefendingSide = right 
            home_team_side = 'right'
        else:
            # home was defending themselves in the left side -> homeTeamDefendingSide = left
            home_team_side = 'left'
    else:  # if home is in offensive zone
        if x_coor > 0:
            # home was attacking on the right side -> homeTeamDefendingSide = left 
            home_team_side = 'left'
        else:
            # home was attacking on the left side -> homeTeamDefendingSide = right 
            home_team_side = 'right'

    df2['homeTeamDefendingSide'] = pd.Series([np.nan] * len(df2), dtype='object')

    for p in range(1, num_period + 1): # why using a loop instead of parallezing: the number of periods is variable 
        period_crn = df2['period'] == p 
        length = sum(period_crn * 1) 
        if p == 1:
            df2.loc[period_crn, 'homeTeamDefendingSide'] = [home_team_side] * length
        else:
            home_team_side = flip(home_team_side)
            df2.loc[period_crn, 'homeTeamDefendingSide'] = [home_team_side] * length
    return df2


def feature_extractor_part_3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts features such as rink_side, shot_distance, shot_angle, and eventOwnerTeamName. 

    Parameters: 
    df

    Returns: 
    df: the input dataframe with extrac columns as `rink_side`, `shot_distance`, `shot_angle`, `eventOwnerTeamName`. 
    """
    df2 = df.copy()
    # finding the rink_side where the puck is heading to. 
    df2['rink_side'] = df2.apply(rink_side_finder, axis=1)
    df2['shot_distance'] = df2.apply(shot_distance_calculator, axis=1)
    df2['shot_angle'] = df2.apply(shot_angle_calculator, axis=1)
    # select the desired cols
    df2['eventOwnerTeamName'] = df2.apply(team_name_finder, axis=1)
    # selected_cols = ['game_id', 'away_team_name', 'home_team_name', 'eventOwnerTeamName', 'shot_distance', 'shot_angle', 'empty_net', 'is_goal']
    return df2

def model_prepartor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the feature engineered dataframe for the machine learning model. 
    1- Scales the columns `shot_distance`, and `shot_angle` of the input dataframe using MinMaxScaler.
    2- Encodes the labels of columns `empty_net` and `is_goal`. 

    Parameters: 
    df: The featured Engineered dataframe 

    Returns: 
    df: a scaled and encoded dataframe
    """
    sc = MinMaxScaler()
    cols = ['shot_distance', 'shot_angle']
    df.loc[:, cols] = sc.fit_transform(df.loc[:, cols].to_numpy().reshape(-1, 2))
    df.loc[:, 'empty_net'] = df.loc[:, 'empty_net'].astype('bool') * 1
    df.loc[:, 'is_goal'] = df.loc[:, 'is_goal'].astype('bool') * 1
    return df
    
def single_data_pipline(game_id: str) -> pd.DataFrame:
    """
    Collects the json file in the `NHL_data` folder with a given game_id, conversts the json file to a csv file, 
    and extracts features. 

    Parameters: 
    game_id (str): the desired game unique identifier 

    Returns: 
    df_extracted (pd.Series): An feature engineeried dataframe 
    """
    desired_game = game_id
    file_path = change_directory()
    data_json = open_json(game_id=desired_game, path=file_path)
    # print(data_json)
    if data_json:
        # tidy the data -> json to dataframe with desired features
        df = tidy_data(desired_game, data_json)
        # save the dataframe in a folder
        # save_csv(game_id=desired_game, df=df, name='tidied')
        # extract and save the desired features
        df1 = feature_extractor_part_1(df)
        if int(desired_game[:4]) <= 2019: # games before 2019, the API does not provide the homeTeamDefendingSide
            df2 = feature_extractor_part_2(df1)
        else: # games above 2019, the API does provide the homeTeamDefendingSide
            df2 = df1
        df_extracted = feature_extractor_part_3(df2)
        # some selected columns for saving the result 
        selected_cols = ['game_id', 'away_team_name', 'home_team_name', 'eventOwnerTeamName', 'shot_distance',
                         'shot_angle', 'empty_net', 'is_goal']
        return df_extracted
    else:
        # if the download was not successful
        raise ValueError(f'The game with id {game_id} does not exist')

def json2csv():
    """
    Gets all json files saved in the `NHL_data` file, and extract features for each, 
    and concates all dataframes into one single dataframe. 
    
    Parameters: 
    None

    Returns: 
    all_df: a dataframe having all information for all games in the `NHL_data`. 
    """
    folder_path = change_directory()
    file_list = os.listdir(folder_path)
    json_file_list = [file for file in file_list if
                      (os.path.isfile(os.path.join(folder_path, file)) and file[-4:] == 'json')]
    dfs = []
    for num, json_file_name in enumerate(json_file_list):
        # looks at each game 
        # take the game_id out
        desired_game = json_file_name[:-5] 
        # extract features for the desired_game 
        df_extracted = single_data_pipline(game_id=desired_game)
        # append the df_extracted for all json files in the `NHL_data` folder.
        dfs.append(df_extracted)

    # concatenate all dataframes into a single dataframe 
    all_df = pd.concat(dfs, axis=0)

    # save the result into  `NHL_data` folder with the name `all_data.csv`
    all_csv_path = os.path.join(folder_path, 'all_data.csv')
    all_df.to_csv(all_csv_path)
    return all_df

def json2df_model(years: list, download_flag=False):
    """
    Downloads the data from the given years into the `NHL_data` if the download_flag is True. 
    Extractes the features for a each game in the folder, and saves all individual dataframes into one integrated dataframe. 

    Parameters: 
    years (list): a list consisting years (int elements) of for which the feature engineering must be done. 

    Returns: 
    a dataframe with selected features 
    """
    # get the folder `NHL_data` 
    
    download_path = change_directory()
    if download_flag: 
        # downaload data
        # download_path = os.path.join("NHL_data")
        
        for year in years:
            obj = downloadData(year, download_path)
            obj.download_nhl_raw_data()
    else: 
        pass 

    folder_path = download_path
    # convert to csv
    all_df = json2csv()

    # the dataframe without the scaling and removing nans 
    all_df.to_csv(os.path.join(folder_path, r'all_df.csv')) 
    
    
    # drop the rows who have missing values in x_coordinate and y_coordinate
    index_c = all_df['shot_distance'].isna() | (all_df['shot_distance'] == '')
    index_d = all_df['shot_angle'].isna() | (all_df['shot_angle'] == '')
    index = index_c | index_d
    # only use the rows who does not have nans or "" 
    df_new = all_df.loc[~index]
    # the dataframe without scaling but with removing nans 
    df_new.to_csv(os.path.join(folder_path, r'all_df_drop.csv')) 
    
    # min-max scaling and label encoding 
    all_df_scaled = model_prepartor(df_new)

    # comment 
    all_df_scaled.to_csv(os.path.join(folder_path, r'all_df_scaled.csv')) 
    
    # save the final result into the `NHL_data` by the name `all_data_final.csv`.
    all_csv_path = os.path.join(folder_path, 'all_data_final.csv')
    
    # choose desired cols
    selected_cols = ['game_id', 'away_team_name', 'home_team_name', 
                     'eventOwnerTeamName', 'shot_distance',
                     'shot_angle', 'empty_net', 'is_goal']
    
    all_df_scaled[selected_cols].to_csv(all_csv_path)
    
    return all_df_scaled

if __name__ == '__main__':
    years = [2016, 2017, 2018, 2019, 2020]
    all_df_final = json2df_model(years)

