# import libraries
import json
import numpy as np
import pandas as pd
import requests
import os
import math
import sklearn
from sklearn.preprocessing import MinMaxScaler


# from preprocessed_modified import downloadData


# functions
# def download_data(game_id:str)-> dict:
#     r"""
#     :param game_id:
#     :return: a dictionary containing the game data with the given game_id
#     """
#     url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
#     try:
#         response = requests.get(url)
#         # raise an error if the response was unsuccessful
#         response.raise_for_status()
#         json_data = response.json()
#         return json_data
#     except Exception as e:
#         print(f'The {e} occurred!')
#         # return None

def tidy_data(game_id: str, data: dict) -> pd.DataFrame:
    r"""
    :param game_id: game_id
    :param data: raw json file
    :return: a dataframe containing desired information
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
    # print(length, len(goal_list), len(period_list), len(periodType_list))
    # print(length, len(x_coordiante_list), len(y_coordiante_list))
    # print(length, len(eventOwnerTeamId_list), len(situationCode_list))
    # print(length, )
    # print(length, len(zoneCode_list))
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
        # 'homeTeamDefendingSide': homeTeamDefendingSide_list,
        'away_team_id': [away_team_id] * length,
        'home_team_id': [home_team_id] * length,
        'away_team_name': [away_team_name] * length,
        'home_team_name': [home_team_name] * length,
        'zoneCode': zoneCode_list,
        'homeTeamDefendingSide': homeTeamDefendingSide_list
    })
    return df


def stength_extractor(row):  # 0541: away (5); home(5) - correct logic
    r"""
    :param row: the row of the dataframe
    :return code: defines the strength of the event w.r.t. to the team who is shooting
    logic: team memeber comparision w.r.t. to the team who is shooting.
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
        code = 'Equal'
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


def empty_net_extractor(row):
    r"""
    :param row: the row of dataframe
    :return : checks whether the net of the team being attacked is empty or not
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


def team_name_finder(row):
    r"""
    :param row: the row of a dataframe
    :param return: the name of the team who is shooting
    """
    home_team_name = row['home_team_name']
    home_team_id = row['home_team_id']
    away_team_id = row['away_team_id']
    away_team_name = row['away_team_name']
    eventOwnerTeamId = row['eventOwnerTeamId']
    if home_team_id == eventOwnerTeamId:
        shooter_team_name = home_team_name  # home is shooting
    elif away_team_id == eventOwnerTeamId:
        shooter_team_name = away_team_name  # away is shooting
    else:
        shooter_team_name = None
    return shooter_team_name


# def rink_side_finder(row):
#     r"""
#     :param row: the row of a dataframe
#     :param return: the rink side of the team who is shooting
#     logic: rink_side is the opposite side of the attacking team's defending side.
#     example:
#     1 - if the home team is shooting, the rink_side = opposite of homeTeamDefendingSide.
#     2 - if the away team is shooting, the rink_side = homeTeamDefendingSide
#     """
#     home_team_id = row['home_team_id']
#     away_team_id = row['away_team_id']
#     # home_team_defending_side = row['homeTeamDefendingSide']
#     zoneCode = row['zoneCode']
#     shooter_id = row['eventOwnerTeamId']
#     x_coor = row['x_coordinate']
#     print(zoneCode, x_coor)
#     if zoneCode == 'O': # offensive
#         if x_coor > 0: # they are shooting to the right
#             rink_side = 'right'
#         if x_coor < 0: # they are shooting to the left
#             rink_side = 'left'
#         else: # we don't know where they are shooting at but it does not change the result for x_net (symmetric)
#             rink_side = 'left'
#     if zoneCode == 'D': # defensive
#         if x_coor > 0: # they are shooting to the left (distant the puck from their defensive zone)
#             rink_side = 'left'
#         if x_coor < 0: # they are shooting to the right
#             rink_side = 'right'
#         else:
#             rink_side = 'right'
#     return rink_side

def rink_side_finder(row):
    r"""
    :param row: the row of a dataframe
    :param return: the rink side of the team who is shooting
    logic: rink_side is the opposite side of the attacking team's defending side.
    example:
    1 - if the home team is shooting, the rink_side = opposite of homeTeamDefendingSide.
    2 - if the away team is shooting, the rink_side = homeTeamDefendingSide
    """
    home_team_id = row['home_team_id']
    away_team_id = row['away_team_id']
    home_team_defending_side = row['homeTeamDefendingSide']
    shooter_id = row['eventOwnerTeamId']
    if home_team_defending_side == 'right':
        home_team_attacking_side = 'left'
    if home_team_defending_side == 'left':
        home_team_attacking_side = 'right'
    else:
        home_team_attacking_side = None
        return None
    if shooter_id == home_team_id:
        # shooter_team = 'home'
        rink_side = home_team_attacking_side  # the rink side of the home offensive zone (=away defensive zone)
    if shooter_id == away_team_id:
        # shooter_team = 'away'
        rink_side = home_team_defending_side  # the rink side of the home defensive zone
    return rink_side


def shot_distance_calculator(row):
    r"""
    :param row: the row of dataframe
    :return : distance which is the distance from which the shot was shooted from the net
    logic: x_net is either (-89) or (89).
    x_net = -89 if the rink_side of the attacking team is on the left.
    x_net = 89 if the rink_side of the attacking team is on the right.
    """
    x = row['x_coordinate']
    y = row['y_coordinate']
    rink_side = row['rink_side']
    if ((rink_side == None) or x == '' or pd.isnull(x) or y == '' or pd.isnull(y)):
        # print(f'--{rink_side}-{pd.isnull(rink_side)}-{x}-{pd.isnull(x)}-{y}-{pd.isnull(y)}')
        return ""
    else:
        # print(f'!!{rink_side}-{pd.isnull(rink_side)}-{x}-{pd.isnull(x)}-{y}-{pd.isnull(y)}')
        # print(rink_side)
        y_net = 0
        if rink_side == 'left':
            x_net = -89
        if rink_side == 'right':
            x_net = 89
        distance = math.sqrt((x - x_net) ** 2 + (y - y_net) ** 2)
        return distance


def shot_angle_calculator(row):
    r"""
    :param row: the row of dataframe
    :return : computed_angle which is the angle from which the shot was shooted from the net
    logic:
    A - if the shot is made in front of the rink (not perpendecular):
    A1 - if the rink_side is on the left: theta =  arctan(|y| / (-89 - |x|))
    A2 - if the rink_side is on the right: theta =  arctan(|y| / (89 - |x|))

    B - if the shot is made behind the net: theta = (not perpendecular): theta = pi/2 + theta
    C - if the shot is perpendecular to the net (x_net = x): theta = pi/2

    D - if the shot is parallel to the vertical netline (y=0): theta = 0
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
    :param df: tided dataframe
    :return: extract features (1) shot_distance, (2) shot_angle
    """
    df2 = df.copy()
    df2['is_goal'] = (df2['goal'] == 'goal') * 1
    df2['home_strength'] = df2.apply(stength_extractor, axis=1)
    df2['empty_net'], df2['is_home'], df2['away_net'], df2['home_net'] = zip(*df2.apply(empty_net_extractor, axis=1))
    return df2


def flip(side):
    if side == 'right':
        flipped = 'left'
    elif side == 'left':
        flipped = 'right'
    else:
        raise ValueError('The side must be either right or left')
    return flipped


def feature_extractor_part_2(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    :param df: tided dataframe
    :return: extract features (1) shot_distance, (2) shot_angle
    """
    df2 = df.copy()
    num_period = df['total_period'][0]  # one of them
    period_crn = df2['period'] == 1
    zone_crn_D = df2['zoneCode'] == 'D'
    zone_crn_O = df2['zoneCode'] == 'O'
    home_team_crn = df2['is_home'] == True
    x_coor_crn = df2['x_coordinate'] != 0
    criterion_D = (period_crn & zone_crn_D & home_team_crn & x_coor_crn)
    criterion_O = (period_crn & zone_crn_O & home_team_crn & x_coor_crn)
    if criterion_D.any():
        df_and = df2.loc[criterion_D, :].reset_index()
        cr_D = True
    if criterion_O.any():
        df_and = df2.loc[criterion_O, :].reset_index()
        cr_D = False
    # print(df_and['x_coordinate'])
    x_coor = df_and.loc[0, 'x_coordinate']
    if cr_D:  # if home is in defensive zone
        if x_coor > 0:
            home_team_side = 'right'
        else:
            home_team_side = 'left'
    else:  # if home is in offensive zone
        if x_coor > 0:
            home_team_side = 'left'
        else:
            home_team_side = 'right'
    df2['homeTeamDefendingSide'] = pd.Series([np.nan] * len(df2), dtype='object')
    for p in range(1, num_period + 1):
        period_crn = df2['period'] == p
        length = sum(period_crn * 1)
        if p == 1:
            df2.loc[period_crn, 'homeTeamDefendingSide'] = [home_team_side] * length
        else:
            home_team_side = flip(home_team_side)
            df2.loc[period_crn, 'homeTeamDefendingSide'] = [home_team_side] * length
    return df2


def feature_extractor_part_3(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    :param df: tided dataframe
    :return: extract features (1) shot_distance, (2) shot_angle
    """
    df2 = df.copy()
    # df2['is_goal'] = (df2['goal'] == 'goal') * 1
    # df2['home_strength'] = df2.apply(stength_extractor, axis = 1)
    # df2['empty_net'], df2['is_home'], df2['away_net'], df2['home_net'] = zip(*df2.apply(empty_net_extractor, axis=1))
    df2['rink_side'] = df2.apply(rink_side_finder, axis=1)
    df2['shot_distance'] = df2.apply(shot_distance_calculator, axis=1)
    df2['shot_angle'] = df2.apply(shot_angle_calculator, axis=1)
    # select the desired cols
    df2['eventOwnerTeamName'] = df2.apply(team_name_finder, axis=1)
    # selected_cols = ['game_id', 'away_team_name', 'home_team_name', 'eventOwnerTeamName', 'shot_distance', 'shot_angle', 'empty_net', 'is_goal']
    return df2


def model_prepartor(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    :param df: the tidied dataframe
    :return: performs min-max scaling on the shot_distance and shot_angle columns
    """
    sc = MinMaxScaler()
    cols = ['shot_distance', 'shot_angle']
    df.loc[:, cols] = sc.fit_transform(df.loc[:, cols].to_numpy().reshape(-1, 2))
    df.loc[:, 'empty_net'] = df.loc[:, 'empty_net'].astype('bool') * 1
    df.loc[:, 'is_goal'] = df.loc[:, 'is_goal'].astype('bool') * 1
    return df


def save_json(game_id, data: dict, path):
    r"""
    :param game_id:
    :param data: json raw downloaded data
    :param path: desired path for the json data
    :return: nothing
    """
    json_object = json.dumps(data, indent=4)
    file_direction = os.path.join(path, f'{game_id}.json')
    with open(file_direction, 'w') as file:
        file.write(json_object)
    pass


def open_json(game_id: str, path):
    file_path = os.path.join(path, fr'{game_id}.json')
    with open(file_path, 'r') as file:
        json_data = json.load(file)
    return json_data


def change_directory(main_folder_name=r'NHL_data'):
    r"""
    :param game_id:
    :return: the desired path of the given game_id to store its files
    """
    current = os.getcwd()
    folder = os.path.join(current, main_folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        pass
    return folder


def save_csv(game_id, df: pd.DataFrame, name):
    folder = change_directory()
    path_csv = os.path.join(folder, f'{game_id}-{name}.csv')
    df.to_csv(path_csv)
    pass


def single_data_pipline(game_id: str) -> pd.DataFrame:
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
        if int(desired_game[:4]) <= 2019:
            df2 = feature_extractor_part_2(df1)
        else:
            df2 = df1
        df_extracted = feature_extractor_part_3(df2)
        # save_csv(game_id=desired_game, df=df_extracted, name='extracted')
        # standardize
        # df_final = model_prepartor(df_extracted)
        selected_cols = ['game_id', 'away_team_name', 'home_team_name', 'eventOwnerTeamName', 'shot_distance',
                         'shot_angle', 'empty_net', 'is_goal']
        # df_selected = df_final.loc[:, selected_cols]
        # return df_selected
        return df_extracted
    else:
        # if the download was not successful
        raise ValueError(f'The game with id {game_id} does not exist')


def json2csv():
    folder_path = change_directory()
    file_list = os.listdir(folder_path)
    json_file_list = [file for file in file_list if
                      (os.path.isfile(os.path.join(folder_path, file)) and file[-4:] == 'json')]
    dfs = []
    for num, json_file_name in enumerate(json_file_list):
        desired_game = json_file_name[:-5]
        df_extracted = single_data_pipline(game_id=desired_game)
        dfs.append(df_extracted)
    all_df = pd.concat(dfs, axis=0)
    all_csv_path = os.path.join(folder_path, 'all_data.csv')
    all_df.to_csv(all_csv_path)
    return all_df


def json2df_model(years: list):
    ## downaload data
    # download_path = os.path.join("NHL_data")
    # for year in available_years:
    #     obj = downloadData(year, download_path)
    #     obj.download_nhl_raw_data()

    # convert to csv
    all_df = json2csv()
    # choose desired cols
    selected_cols = ['game_id', 'away_team_name', 'home_team_name', 'eventOwnerTeamName', 'shot_distance',
                     'shot_angle', 'empty_net', 'is_goal']
    # drop the rows who have missing values in x_coordinate and y_coordinate
    index_c = all_df['shot_distance'].isna() | (all_df['shot_distance'] == '')
    index_d = all_df['shot_angle'].isna() | (all_df['shot_angle'] == '')
    index = index_c | index_d
    df_new = all_df.loc[~index]
    # # min-max scaling
    all_df_final = model_prepartor(df_new)

    folder_path = change_directory()
    all_csv_path = os.path.join(folder_path, 'all_data_final.csv')
    all_df_final[selected_cols].to_csv(all_csv_path)
    return all_df_final[selected_cols]


if __name__ == '__main__':
    # all_df = json2csv()
    # print(df.head())
    years = [2016, 2017, 2018, 2019, 2020]
    all_df_final = json2df_model(years)