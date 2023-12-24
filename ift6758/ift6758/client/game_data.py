import os
import requests
import json
import pandas as pd
import numpy as np


def get_game_data(game_id, NHL_API_BASE_URL='https://api-web.nhle.com/v1/gamecenter'):
    """
    This function downloads NHL game data for a specified game ID and saves it to a JSON file.

    Parameters:
    - game_id (int): The NHL game ID to download data for.
    - NHL_API_BASE_URL (str): The base URL of the NHL API.
    """
    game_url = f"{NHL_API_BASE_URL}/{game_id}/play-by-play"

    response = requests.get(game_url)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Failed to fetch data for game {game_id}")
        return None


def calculate_angle(x: float, y: float, offensive_side: int) -> float:
    """
    Helper function. Computes and returns the angle between the xy shooter coordinates
    and the net they are scoring into based on "home_offensive_side" relative to the center line of the net.
    """
    goal_coord = np.array([89, 0])
    if x is None or y is None:
        return 0
    goal_coord = offensive_side * goal_coord

    relative_x = x - goal_coord[0]  # bring x-coordinate relative to the goal
    angle = 0  # Defaults to 0 if x = [-89 or 89]. That's actually common.
    y += 1e-5  # avoid division by zero
    if np.sign(goal_coord[0]) == -1:  # left goal
        if (np.sign(relative_x)) == 1:  # front of the goal
            angle = np.arctan(np.abs(y) / relative_x)
        elif (np.sign(relative_x)) == -1:  # behind the goal
            angle = np.arctan(np.abs(relative_x) / y) + np.pi / 2  # +90 degrees to account its from behind
    elif np.sign(goal_coord[0]) == 1:  # right goal
        if (np.sign(relative_x)) == -1:  # front of the goal
            angle = np.arctan(np.abs(y) / np.abs(relative_x))
        elif (np.sign(relative_x)) == 1:  # behind the goal
            angle = np.arctan(relative_x / y) + np.pi / 2  # +90 degrees to account its from behind
    return np.rad2deg(angle)


def calculate_distance(x, y, offensive_side):
    """
    Computes and returns the distance between the xy shooter coordinates
    and the net they are scoring into based on "home_offensive_side"..
    """
    goal_coord = np.array([89, 0])
    if x is None or y is None:
        return None
    goal_coord = offensive_side * goal_coord
    return np.linalg.norm(np.array([x, y]) - goal_coord)


def get_features(data):
    game_id = data['id']
    away_team_id, away_team_name = data['awayTeam']['id'], data['awayTeam']['name']['default']
    home_team_id, home_team_name = data['homeTeam']['id'], data['homeTeam']['name']['default']
    period, time_remaining, home_goals, away_goals, reqd_all_play_df = None, None, None, None, None

    all_plays = data['plays']
    home_goals, away_goals = 0, 0
    reqd_all_play_list = []
    for play in all_plays:
        period = play['period']
        time_remaining = play['timeRemaining']

        if play['typeDescKey'] == 'goal':
            is_goal = 1
            home_goals, away_goals = play['details']['homeScore'], play['details']['awayScore']
            x_coord, y_coord = play['details']['xCoord'], play['details']['yCoord']
            if play['details']['eventOwnerTeamId'] == home_team_id:
                team = home_team_name
                if int(play['situationCode'][0]) == 1:
                    is_emptynet = 0
                else:
                    is_emptynet = 1
            else:
                team = away_team_name
                if int(play['situationCode'][3]) == 1:
                    is_emptynet = 0
                else:
                    is_emptynet = 1
            goal_distance = calculate_distance(x_coord, y_coord, np.sign(x_coord))
            shot_angle = calculate_angle(x_coord, y_coord, np.sign(x_coord))
            reqd_all_play_list.append([team, x_coord, y_coord, goal_distance, shot_angle, is_goal, is_emptynet])

        elif play['typeDescKey'] == 'shot-on-goal':
            is_goal, is_emptynet = 0, 0
            x_coord, y_coord = play['details']['xCoord'], play['details']['yCoord']
            goal_distance = calculate_distance(x_coord, y_coord, np.sign(x_coord))
            shot_angle = calculate_angle(x_coord, y_coord, np.sign(x_coord))
            if play['details']['eventOwnerTeamId'] == home_team_id:
                team = home_team_name
            else:
                team = away_team_name
            reqd_all_play_list.append([team, x_coord, y_coord, goal_distance, shot_angle, is_goal, is_emptynet])
    reqd_all_play_df = pd.DataFrame(reqd_all_play_list,
                                    columns=['team', 'x_coord', 'y_coord', 'shot_distance', 'shot_angle', 'is_goal',
                                             'is_emptynet'])
    return game_id, home_team_name, away_team_name, period, time_remaining, home_goals, away_goals, reqd_all_play_df