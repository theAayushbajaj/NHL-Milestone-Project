# import libraries
import json
import numpy as np
import pandas as pd
import requests
import os
import math
import sklearn
from sklearn.preprocessing import MinMaxScaler


# functions
def download_data(game_id: str) -> dict:
    r"""
    :param game_id:
    :return: a dictionary containing the game data with the given game_id
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


def tidy_data(data: dict) -> pd.DataFrame:
    r"""
    :param data: raw json file
    :return: a dataframe containing desired information
    """
    away_team_id = data['awayTeam']['id']
    away_team_name = data['awayTeam']['name']['default']

    home_team_id = data['homeTeam']['id']
    home_team_name = data['homeTeam']['name']['default']

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

    length = len(Plays)
    df = pd.DataFrame({
        'season': [desired_game[:4]] * length,
        'game_id': [desired_game] * length,
        'goal': goal_list,
        'period': period_list,
        'periodType': periodType_list,
        'x_coordinate': x_coordiante_list,
        'y_coordinate': y_coordiante_list,
        'eventOwnerTeamId': eventOwnerTeamId_list,
        'situationCode': situationCode_list,
        'homeTeamDefendingSide': homeTeamDefendingSide_list,
        'away_team_id': [away_team_id] * length,
        'home_team_id': [home_team_id] * length,
        'away_team_name': [away_team_name] * length,
        'home_team_name': [home_team_name] * length
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
    else:
        home_team_attacking_side = 'right'
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
    if rink_side == "right":
        x_net = 89
    else:
        x_net = -89

    x_dist_abs, y_dist_abs = np.abs(x_net - x), np.abs(y)
    shot_taken_behind_net = (x_net == 89 and x > 89) or (x_net == -89 and x < -89)
    shot_taken_perpendicular_net = (x_net == x)

    if (y == 0):
        angle = 0
    else:
        if (shot_taken_perpendicular_net):
            angle = np.pi / 2
        else:
            angle = np.arctan(y_dist_abs / x_dist_abs)
            if (shot_taken_behind_net):
                angle += np.pi / 2
    computed_angle = round(np.rad2deg(angle))
    return computed_angle


def feature_extractor(df: pd.DataFrame) -> pd.DataFrame:
    r"""
    :param df: tided dataframe
    :return: extract features (1) shot_distance, (2) shot_angle
    """
    df2 = df.copy()
    df2['is_goal'] = (df2['goal'] == 'goal') * 1
    df2['home_strength'] = df2.apply(stength_extractor, axis=1)
    df2['empty_net'], df2['is_home'], df2['away_net'], df2['home_net'] = zip(*df2.apply(empty_net_extractor, axis=1))
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
    scaler_shot_distance = MinMaxScaler()
    scaler_shot_angle = MinMaxScaler()
    cols = ['shot_distance', 'shot_angle']
    df[cols[0]] = scaler_shot_distance.fit_transform(df[cols[0]].to_numpy().reshape(-1, 1))
    df[cols[1]] = scaler_shot_angle.fit_transform(df[cols[1]].to_numpy().reshape(-1, 1))
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


def change_directory(game_id: str):
    r"""
    :param game_id:
    :return: the desired path of the given game_id to store its files
    """
    current = os.getcwd()
    folder = os.path.join(current, f'{game_id}')
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        pass
    return folder


def save_csv(game_id, df: pd.DataFrame, name):
    folder = change_directory(game_id)
    path_csv = os.path.join(folder, f'{game_id}-{name}.csv')
    df.to_csv(path_csv)
    pass


def data_pipline(game_id: str) -> pd.DataFrame:
    desired_game = game_id
    data_json = download_data(game_id=desired_game)
    if data_json:
        # save the json file in a folder
        file_path = change_directory(game_id=desired_game)
        save_json(game_id=desired_game, data=data_json, path=file_path)
        # tidy the data -> json to dataframe with desired features
        df = tidy_data(data_json)
        # save the dataframe in a folder
        # save_csv(game_id=desired_game, df=df, name='tidied')
        # extract and save the desired features
        df_extracted = feature_extractor(df)
        # save_csv(game_id=desired_game, df=df_extracted, name='extracted')
        # standardize
        df_final = model_prepartor(df_extracted)
        selected_cols = ['game_id', 'away_team_name', 'home_team_name', 'eventOwnerTeamName', 'shot_distance',
                         'shot_angle', 'empty_net', 'is_goal']
        df_selected = df_final.loc[:, selected_cols]
        save_csv(game_id=desired_game, df=df_selected, name='final')
        return df_selected
    else:
        # if the download was not successful
        raise ValueError(f'The game with id {game_id} does not exist')


if __name__ == '__main__':
    desired_game = '2021030112'
    df = data_pipline(game_id=desired_game)
    # print(df.head())