import requests
import os
import json
import time
from ift6758.ift6758.client.game_data import get_features
import datetime

PATH_FOR_LIVE_DATA = os.path.join(os.path.dirname(__file__), 'live_data_storage')

class HockeyGameTracker:
    def __init__(self, id_of_game: int):
        self.processed_data = set()
        self.directory_path = PATH_FOR_LIVE_DATA
        
        self.game_file = os.path.join(self.directory_path, f'{int(id_of_game)}.json')
        os.makedirs(os.path.dirname(self.game_file), exist_ok=True)
        self.game_id = id_of_game

    def check_game_status(self, game_data):
        plays_list = game_data['plays']
        data_counter = 0
        for action in plays_list:
            game_period = action['period']
            remaining_time = action['timeRemaining']

            if action['typeDescKey'] in ['goal', 'shot-on-goal']:
                data_counter += 1
        game_status = 'finished' if game_period == 3 and remaining_time == '00:00' else 'live'
        return game_status, data_counter

    def retrieve_game_status(self):
        file_exists = os.path.isfile(self.game_file)
        if not file_exists:
            return 'ping1', None, None
        with open(self.game_file, 'r+') as file:
            loaded_data = json.load(file)
        team_info = self.parse_team_data(loaded_data)
        game_status, data_counter = self.check_game_status(loaded_data)
        return game_status, *team_info

    def parse_team_data(self, data):
        team_data = data['plays']
        away_info = (data['awayTeam']['id'], data['awayTeam']['name']['default'])
        home_info = (data['homeTeam']['id'], data['homeTeam']['name']['default'])
        return away_info, home_info

    def fetch_and_store_data(self):
        API_URL = f"https://api-web.nhle.com/v1/gamecenter/{self.game_id}/play-by-play"
        api_response = requests.get(API_URL)
        with open(self.game_file, 'w') as file:
            file.write(api_response.text)
        with open(self.game_file, 'r+') as file:
            game_details = json.load(file)
            return extract_game_data(game_details)

    def update_live_game(self):
        file_exists = os.path.isfile(self.game_file)
        if not file_exists:
            return self.fetch_and_store_data(), False
        else:
            with open(self.game_file, 'r+') as file:
                current_game = json.load(file)
                current_status, index = self.check_game_status(current_game)
            game_info = self.fetch_and_store_data()
            if current_status == 'live':
                game_info[-1] = game_info[-1].loc[game_info[-1].index >= index].reset_index(drop=True)
            return game_info, False

    def periodic_api_check(self, duration: int = 120, interval: float = 60.0) -> list:        
        collected_data = []
        start_time = time.time()
        while (time.time() - start_time) < duration:
            collected_data.extend(self.update_live_game())
            time.sleep(interval - ((time.time() - start_time) % interval))
        return collected_data
    
