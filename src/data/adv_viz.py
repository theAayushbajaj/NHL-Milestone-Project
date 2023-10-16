import pandas as pd

def game_number(game_df):
    list_teams = list(game_df['team shot'].unique())
    games_per_team = {}
    for team in list_teams:
        new_game_df = game_df[game_df['team shot'] == team]
        games_per_team[team] = new_game_df['game id'].unique().shape[0]
    return games_per_team

def agg_loc(agg_df, games_per_team):
    agg_df['y'] = agg_df['y_transformed'] * (-1)
    y_bins, goal_dist_bins = list(range(-41, 42, 4)), list(range(0, 94, 4))
    agg_df['y_bins'], agg_df['goal_dist_bins'] = pd.cut(agg_df['y'], y_bins), pd.cut(agg_df['goal_dist'], goal_dist_bins)
    new_agg_df = agg_df.groupby(['season', 'team shot', 'y_bins', 'goal_dist_bins'])['goal'].size().to_frame('total').reset_index()
    new_agg_df['games_per_team'] = new_agg_df['team shot'].apply(lambda x: games_per_team.get(x))
    new_agg_df['average_per_hour'] = new_agg_df['total'] / new_agg_df['games_per_team']
    new_agg_df['y_mid'] = new_agg_df['y_bins'].apply(lambda x: (x.left + x.right) / 2)
    new_agg_df['goal_mid'] = new_agg_df['goal_dist_bins'].apply(lambda x: (x.left + x.right) / 2)
    return new_agg_df

def agg_shot(df):
    df['y'] = df['y_transformed'] * (-1)
    total_games = df['game id'].unique().shape[0]
    y_bins, goal_dist_bins = list(range(-41, 42, 4)), list(range(0, 94, 4))
    df['y_bins'], df['goal_dist_bins'] = pd.cut(df['y'], y_bins), pd.cut(df['goal_dist'], goal_dist_bins)
    new_df = df.groupby(['season', 'y_bins', 'goal_dist_bins'])['goal'].size().to_frame('total').reset_index()
    new_df['average_per_hour'] = new_df['total'] / (2*total_games)
    new_df['y_mid'] = new_df['y_bins'].apply(lambda x: (x.left + x.right) / 2)
    new_df['goal_mid'] = new_df['goal_dist_bins'].apply(lambda x: (x.left + x.right) / 2)
    return new_df

def trans_coord(rinkSide, coordinate):
    if rinkSide == "right": return (-1) * coordinate
    else: return coordinate

def trans_col(col_df):
    col_df = col_df.copy()
    col_df[['coordinates_x', 'coordinates_y']] = col_df.coordinates.str.split(", ", expand=True,)
    col_df['coordinates_x'] = col_df['coordinates_x'].str.replace('\[', '', regex=True)
    col_df['coordinates_x'] = col_df['coordinates_x'].str.replace(',', '', regex=True)
    col_df['coordinates_y'] = col_df['coordinates_y'].str.replace(']', '', regex=True)
    col_df[['coordinates_x', 'coordinates_y']] = col_df[['coordinates_x', 'coordinates_y']].apply(pd.to_numeric)
    col_df['x_transformed'] = col_df.apply(lambda x: trans_coord(x['rinkSide'], x['coordinates_x']), axis=1)
    col_df['y_transformed'] = col_df.apply(lambda x: trans_coord(x['rinkSide'], x['coordinates_y']), axis=1)
    col_df = col_df.drop(col_df[(col_df.x_transformed < 25) & (col_df.x_transformed > 89)].index)
    col_df['goal_dist'] = col_df.apply(lambda x: (89 - x['x_transformed']), axis=1)
    return col_df

def season_aggregation(center_y, center_goal, league_df):
    league = league_df.loc[(league_df["y_mid"] == center_y) & (league_df["goal_mid"] == center_goal), 'average_per_hour']
    return league.iloc[0]

def main():
    input_data = pd.read_csv('complex_visuals.csv')
    split_df = input_data.dropna(subset=['rinkSide']).copy()
    season_list = [2016, 2017, 2018, 2019, 2020, 2021]
    major_df = pd.DataFrame()
    for season_year in season_list:
        df = trans_col(split_df[split_df["season"] == season_year])
        games_per_team = game_number(df)
        df_league = agg_shot(df)
        nhl_team_df = agg_loc(df, games_per_team)
        nhl_team_df['league_average'] = nhl_team_df.apply(lambda x: season_aggregation(x['y_mid'], x['goal_mid'], df_league), axis=1)
        nhl_team_df['raw_diff'] = nhl_team_df['average_per_hour'] - nhl_team_df['league_average']
        res_df = pd.concat([major_df, nhl_team_df], ignore_index=True)
    res_df.to_csv('complex_diff.csv', index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    main()
