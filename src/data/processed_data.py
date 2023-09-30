import json
import os
import urllib.request
import sys
from tqdm import tqdm
from urllib.error import HTTPError


class downloadData:
    r"""
    Class to download NHL Hockey Data
    """

    def __init__(self, target_year: str, data_dir_path: str) -> None:
        r"""
        Args:
            target_year (str): The year that we want to get data
            data_dir_path (str): Path to the directory we want to store data (not including year)
        """

        self.target_year = target_year
        self.data_dir_path = data_dir_path

    def download_nhl_data(self, path: str, nhl_game_id: str) -> None:
        r"""
        Download NHL play-by-play data of a specific game into a particular directory path

        Args:
            path: Path to the directory
            nhl_game_id: Game ID of the NHL game that we want to download the data of
        """

        file_path = os.path.join(path, nhl_game_id + ".json")
        
        # Return if file path already exists
        if(os.path.exists(file_path)):
                return
            
        try:
            # Read NHL play-by-play data for both regular season and playoffs game settings
            with urllib.request.urlopen("https://statsapi.web.nhl.com/api/v1/game/" + nhl_game_id + "/feed/live/") as url:
                data = json.load(url)
                if ("messageNumber" in data and "message" in data 
                    and data["messageNumber"] == 2 and data["message"] == "Game data couldn't be found"):
                    pass
                else:
                    with open(file_path, 'w') as outfile:
                        json.dump(data, outfile)
        except HTTPError as he:
            print(nhl_game_id)
            print(he.reason)
        except Exception:
            print('nhl_game_id: '+str(nhl_game_id))
            e_type, e_value, e_traceback = sys.exc_info()
            print(e_value)

    def download_nhl_raw_data(self) -> None:
        r"""
        Function to extract NHL games' data for a specific year
        """

        accessible_years = ['2016', '2017', '2018', '2019', '2020', '2021']
        
        if(self.target_year not in accessible_years):
            print("Dataset does not contain the entered year")
            return
            
        # Initializing regular and playoff settings' path
        regular_dir_path = os.path.join(self.data_dir_path, self.target_year, 'regular_games')
        playoff_dir_path = os.path.join(self.data_dir_path, self.target_year, 'playoff_games')
        
        # Sanity check for directories's existence
        if not os.path.exists(regular_dir_path):
            os.makedirs(regular_dir_path)
        if not os.path.exists(playoff_dir_path):
            os.makedirs(playoff_dir_path)
        
        # Download data of regular games for the selected year
        print(f'Downloading regular games data for {self.target_year}...')
        
        # Year 2016 has 1230 games, while the remaining available years have data of 1270 games
        ID_range = 1231 if (self.target_year=='2016') else 1271
        
        for ID in tqdm(range(1, ID_range)):
            # Convert ID from integer to string
            ID_str =  "0" * (4 - len(str(ID))) + str(ID)
            regular_game_id = self.target_year + "02" + ID_str
            
            # Download data of each game
            self.download_nhl_data(regular_dir_path, regular_game_id)
        
        # Download data of playoff games for the selected year
        print(f"Downloading playoff games data for {self.target_year}...")
        
        for round in tqdm(range(1, 5)):
            # `round 1` comprises `8` game matchups, `round 2` has game matchups of `4` and so on
            matchups = int(2**(3 - round))
            for matchup_number in range(1, matchups + 1):
                # Each match up has 7 games in total
                for game_id in range(1, 8):
                    playoff_game_id = self.target_year + "030" + str(round) + str(matchup_number) + str(game_id)
                    self.download_nhl_data(playoff_dir_path, playoff_game_id)

download_path = os.path.join("raw_data")
available_years = ['2016', '2017', '2018', '2019', '2020', '2021']
for year in available_years:
    obj = downloadData(year, download_path)
    obj.download_nhl_raw_data()
