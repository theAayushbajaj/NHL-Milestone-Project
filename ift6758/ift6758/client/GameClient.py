import requests
import os
import json
import time
from .game_data import get_features
import datetime

LIVE_RAW_DATA_PATH = os. getcwd()


class GameClient:
    def __init__(self, game_id: int):
        self.processed_events = set()
        self.save_path = LIVE_RAW_DATA_PATH

        self.file_path = os.path.join(self.save_path, f'{int(game_id)}.json')
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.game_id = game_id

    def get_status(self, data):
        all_plays = data['plays']
        df_index = 0
        for play in all_plays:
            period = play['period']
            time_remaining = play['timeRemaining']

            if play['typeDescKey'] == 'goal' or play['typeDescKey'] == 'shot-on-goal':
                df_index += 1
        if period == 3 and time_remaining == '00:00':
            status = 'finished'
        else:
            status = 'live'
        return status, df_index

    def get_status_game(self):
        print(self.file_path)
        data_exist = os.path.isfile(self.file_path)
        if not data_exist:
            return 'ping1', None, None
        with open(self.file_path, 'r+') as f:
            data = json.load(f)
        # print(data)
        period, time_remaining = None, None
        all_plays = data['plays']
        away_team_id, away_team_name = data['awayTeam']['id'], data['awayTeam']['name']['default']
        home_team_id, home_team_name = data['homeTeam']['id'], data['homeTeam']['name']['default']
        df_index = 0
        for play in all_plays:
            period = play['period']
            time_remaining = play['timeRemaining']

            if play['typeDescKey'] == 'goal' or play['typeDescKey'] == 'shot-on-goal':
                df_index += 1
        if period is None:
            status = 'no_game'
        if period == 3 and time_remaining == '00:00':
            status = 'finished'
        else:
            status = 'live'
        return status, home_team_name, away_team_name

    def get_reqd_df(self):

        NHL_API_BASE_URL = 'https://api-web.nhle.com/v1/gamecenter'
        game_url = f"{NHL_API_BASE_URL}/{self.game_id}/play-by-play"
        response = requests.get(game_url)
        with open(self.file_path, 'w') as f:
            f.write(response.text)
        with open(self.file_path, 'r+') as f:
            game = json.load(f)
            game_id, home_team_name, away_team_name, period, time_remaining, home_goals, away_goals, reqd_all_play_df = get_features(
                game)
            return game_id, home_team_name, away_team_name, period, time_remaining, home_goals, away_goals, reqd_all_play_df

    def process_live_game(self):
        # Fetch live game data
        no_change_flag = False
        data_exist = os.path.isfile(self.file_path)
        if not data_exist:
            game_id, home_team_name, away_team_name, period, time_remaining, home_goals, away_goals, reqd_all_play_df = self.get_reqd_df()
            return game_id, home_team_name, away_team_name, period, time_remaining, home_goals, away_goals, reqd_all_play_df, no_change_flag
        else:
            with open(self.file_path, 'r+') as f:
                game = json.load(f)
                status, df_index = self.get_status(game)

            game_id, home_team_name, away_team_name, period, time_remaining, home_goals, away_goals, reqd_all_play_df = self.get_reqd_df()
            # print(df_index)
            # print(reqd_all_play_df.tail())
            if status == 'live':
                # if len(reqd_all_play_df)==df_index:
                #     no_change_flag = True
                # else:
                reqd_all_play_df = reqd_all_play_df.loc[reqd_all_play_df.index >= df_index].reset_index(drop=True)
            return game_id, home_team_name, away_team_name, period, time_remaining, home_goals, away_goals, reqd_all_play_df, no_change_flag

    def timed_ping_api(self, time_to_run: int = 120.0, time_interval=60.0) -> list:

        new_events = []
        starttime = time.time()
        while (time.time() - starttime) < time_to_run:
            new_events.extend(self.process_live_game())
            time.sleep(time_interval - ((time.time() - starttime) % time_interval))
        return new_events


# # Example usage
# if __name__ == '__main__':
#     game_id_example = 2022030411
#     game_client = GameClient(game_id_example)
#     game_client.timed_ping_api()
