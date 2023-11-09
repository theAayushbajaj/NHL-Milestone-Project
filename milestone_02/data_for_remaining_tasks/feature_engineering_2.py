import pandas as pd
import numpy as np

def compute_shot_distance(x_shot, y_shot, rink_side):
    """
    A supporting function to calculate distance between the shot and the net goal post
    :param x_shot: shot's x coordinate
    :param y_shot: shot's y coordinate
    :param rink_side: net side, only "left" or "right" are valid
    :return: the distance of the shot
    """
    if(rink_side not in ['left','right']): return ""
    net_coords, shot_coords = np.array([89, 0]) if rink_side=="right" else np.array([-89, 0]), np.array([x_shot, y_shot])
    shot_distance = round(np.linalg.norm(net_coords - shot_coords))
    return shot_distance
    

def add_shot_distance_and_correct_coordinates(df_game):
    """
    Calculate the shot distance and correct the wrong coordinates recorded
    :param df_game: The tidied dataframe
    :return: A dataframe with correct coordinates and the newly incorporated "shot distance" feature
    """
    df_shot_distance = df_game.copy()
    np_shot_distance = df_shot_distance.apply(lambda game: compute_shot_distance(
                                                    game['x shot'], game['y shot'], game['rinkSide']),
                                                    axis=1)
    
    inverted_shot_distance = df_shot_distance.apply(lambda game: compute_shot_distance(
                                                    -(game['x shot']), game['y shot'], game['rinkSide']),
                                                    axis=1)
    
    if(np.mean(inverted_shot_distance) < np.mean(np_shot_distance)):
        df_shot_distance['shot distance'] = inverted_shot_distance
        # Presence of incorrect information, hence inverting the rinkSide
        df_shot_distance['rinkSide'] = df_shot_distance['rinkSide'].apply(lambda side: "right" if side=="left" else "left")
    else:
        df_shot_distance['shot distance'] = np_shot_distance
    
    return df_shot_distance

def compute_shot_angle(x_shot, y_shot, rink_side):
    """
    Determine the angle of the shot relative to the net
    :param shot_coordinates: A numpy array indicating the shot coordinates (e.g., [46, 25])
    :param net_coordinates: A numpy array indicating the net coordinates (e.g., [-89, 0])
    :return: The degree measurement of the shot angle from the net
    """
    if(rink_side not in ['left','right'] or np.isnan(x_shot) or np.isnan(y_shot)):
        return ""
    
    if rink_side=="right":
        x_net_coord = 89
    else:
        x_net_coord = -89
    
    x_dist_abs, y_dist_abs = np.abs(x_net_coord - x_shot), np.abs(y_shot)    
    shot_taken_behind_net = (x_net_coord==89 and x_shot>89) or (x_net_coord==-89 and x_shot<-89)
    shot_taken_perpendicular_net = (x_net_coord==x_shot)
    
    if(y_shot == 0): angle = 0
    else:
        if(shot_taken_perpendicular_net): angle = np.pi/2
        else:
            angle = np.arctan(y_dist_abs/x_dist_abs)
            if(shot_taken_behind_net): angle += np.pi/2
    computed_angle = round(np.rad2deg(angle))

    return computed_angle


def add_game_shot_angle(df_game):
    """
    A function to include a new column labeled "shot angle" in the dataframe, representing the angle of the shot
    :param df_game: The tidied dataframe
    :return: A dataframe containing the newly included "shot angle" feature
    """
    df_add_shot_angle = df_game.copy()
    df_add_shot_angle['shot angle'] = df_add_shot_angle.apply(lambda game: compute_shot_angle(
                                                                game['x shot'], game['y shot'], game['rinkSide']),
                                                                axis=1)
    
    return df_add_shot_angle

def compute_distance_previous_current_events(x_shot, y_shot, x_last, y_last):
    """
    A utility function designed to compute the distance between two events while automatically disregarding inappropriate values
    :param x_shot: shot's x coordinate
    :param y_shot: shot's y coordinate
    :param x_last: previous game's x coordinate
    :param y_last: previous game's y coordinate
    :return: The differnce between the current shot and the last game event
    """
    if(np.isnan(x_last) or np.isnan(y_last)): return ""
    else:
        coord_shot, coord_last = np.array([x_shot, y_shot]), np.array([x_last, y_last])
        return round(np.linalg.norm(coord_shot - coord_last))

def compute_speed(distance_from_last_event, time_from_last_event):
    """
    A helper function to compute the shot speed while automatically disregarding inappropriate values
    :param distance_from_last_event: The shot's distance from the preceding game event
    :param time_from_last_event: The time duration between the previous event and the present shot
    :return: The shot speed
    """
    if(distance_from_last_event != "" and time_from_last_event != 0):
        return round(distance_from_last_event / time_from_last_event)
    else: return ""

def add_distance_from_last_event_and_speed(df_game):
    """
    A function to include two new features ("distance from last event" and "speed") in the dataframe
    :param df_game: The dataframe
    :return: The dataframe containing the aforementioned two new features
    """
    df_add_speed_distance_features = df_game.copy()
    df_add_speed_distance_features['distance from last event'] = df_add_speed_distance_features.apply(lambda game: compute_distance_previous_current_events(
                                                                game['x shot'], game['y shot'], game['x last event'],
                                                                game['y last event']), axis=1)
    df_add_speed_distance_features['speed'] = df_add_speed_distance_features.apply(lambda game:compute_speed(
                                                                game['distance from last event'], game['time from last event']),
                                                                axis=1)
    return df_add_speed_distance_features


def compute_change_in_angle(is_rebound, current_shot_angle, y_shot, x_last, y_last, rinkSide):
    """
    A helper function for computing the difference in angle between two consecutive shots in a rebound
    :param is_rebound: A binary variable to indicate if the shot is a rebound or not
    :param current_shot_angle: Current shot's angle
    :param y_shot: shot's y coordinate
    :param x_last: previous shot's x coordinate
    :param y_last: previous shot's y coordinate
    :param rinkSide: net side which is used to compute the angle of the preceding shot
    :return: The difference in angle for a rebound
    """
    if(not is_rebound): return 0
    
    last_shot_angle = compute_shot_angle(x_last, y_last, rinkSide)
    if(last_shot_angle==""): return 0
    
    # If two vertical shots were taken, subtract current from last vertical coordinates
    if(np.sign(y_shot)==np.sign(y_last)): return np.abs(last_shot_angle - current_shot_angle)
    else: return (last_shot_angle+current_shot_angle)

def add_change_in_game_shot_angle(df_game):
    """
    A utility function to include the feature named "change in shot angle" indicating The difference in angle for a rebound
    :param df_game: The dataframe containing already included new feature column named "shot angle"
    :return: The dataframe containing the aforementioned new feature column labeled "change in shot angle"
    """
    df_change_in_shot_angle = df_game.copy()
    df_change_in_shot_angle['change in shot angle'] = df_change_in_shot_angle.apply(lambda game: compute_change_in_angle(
                                                                            game['is rebound'], game['shot angle'],
                                                                            game['y shot'], game['x last event'],
                                                                            game['y last event'], game['rinkSide']), axis=1)
    return df_change_in_shot_angle


def feature_engg(df_tidy_game):
    """
    The main function for the task "feature engineering"
    :param df_tidy_game: The tidied version of nhl data, stored in a dataframe
    :return: A new dataframe incorporates all the essential new game features.
    """
    nhl_game_df = df_tidy_game.copy()
    
    print("Start feature engg for the tidy game data...")
    
    print("Start correcting game coordinates and adding shot distance...")

    # Calling this function is imperative at first so that all the pre-exising incorrect coordinates can be corrected
    nhl_game_df = add_shot_distance_and_correct_coordinates(nhl_game_df)

    print("Start incorporating the shot angle for each game...")
    nhl_game_df = add_game_shot_angle(nhl_game_df)

    print("Start including difference in the shot angle for each game...")
    nhl_game_df = add_change_in_game_shot_angle(nhl_game_df)

    print("Start adding the disparity between the last event and the speed of the shot...")
    nhl_game_df = add_distance_from_last_event_and_speed(nhl_game_df)

    print("Finishing up the feature engineering process!!!")
    return nhl_game_df
    
df_nhl_data = pd.read_csv("df_tidy_data.csv")
df_nhl_data = feature_engg(df_nhl_data)
df_nhl_data['empty net'] = df_nhl_data['empty net'].map({True: 1, False: 0})
df_nhl_data.to_csv('df_data.csv',index=False)