---
layout: post
title: Milestone 1
---

### Task 1 - Data Acquisition

**Download NHL Data**

In this project, we are given NHL Hockey dataset. The original curated dataset can be accessed through REST API endpoint: https://statsapi.web.nhl.com/api/v1/game/[GAME_ID]/feed/live/. Moreover, 
`GAME_ID` is of `10` digits long. The first 4 represent game's season (e.g. 2017, 2019), the next 2 represent the type of game played (01 = preseason, 02 = regular season, 03 = playoffs, 04 = all-star), and the remaining 4 allude a particular season game.

Given dataset also comprises two game types, i.e. regular and playoff games, which have the following dataset characteristics:
(i) regular seasons: the last 4 digits of `GAME_ID` ranges from 0001 to 1270, which is merely applicable to those seasons with 31 teams, and varies between 0001 and 1230 for those with 30 teams;
(ii) playoff games: the last three digits of `GAME_ID` represent the round, matchup, and game number, respectively.

To acquire (download) the given dataset, we create a class `downloadData` that has two monumental modules:

(a) `download_nhl_raw_data` is responsible to download all the NHL data in accordance with a selected season (or year), given the parent directory path `data_dir_path`. The class `downloadData` receives the user inputs and sets its attributes inside its `constructor` as follows:

```python
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
```

Owing to the limitation of data for avaialble seasons, the implemented approach lets the user to download data for five consective seasons, i.e. from 2016-17 to 2020-21.

```python
	def download_nhl_raw_data(self) -> None:
		r"""
		Function to extract NHL games' data for a specific year
		"""

		accessible_years = ['2016', '2017', '2018', '2019', '2020', '2021']
		
		if(self.target_year not in accessible_years):
			print("Dataset does not contain the entered year")
```

Here, the user-selected year is being matched with the available years for validity. Once the year's validity has been confirmed, we initialize paths for the two given game types with the arguments provided, following which essential directories are created in case they do not exist already.

```python
		# Initializing regular and playoff settings' path
		regular_dir_path = os.path.join(self.data_dir_path, self.target_year, 'regular_games')
		playoff_dir_path = os.path.join(self.data_dir_path, self.target_year, 'playoff_games')

		# Sanity check for directories's existence
		if not os.path.exists(regular_dir_path):
			os.makedirs(regular_dir_path)
		if not os.path.exists(playoff_dir_path):
			os.makedirs(playoff_dir_path)
```

The following snippet is to download all the relevant data files for `regular` game type for the user-provided target year (season):

```python
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
```

It is worth mentioning that for 2016, we have 30 teams with total of 1230 games, whereas for each season following 2016, total of 1270 games were played with engagement of 31 teams.

```python
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
```				

(b) We also create a utility function named `download_nhl_data`	that is responsible for downloading a specific game's data from the provided NHL Data API given `NHL Game ID`.

```python
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
```
				
Voila!! All the necessary functions that could aid in downloading and structuring the NHL data have been prepared. Running the following short code snippet will collect and arrange the data of all seasons (starting 2016 to 2021) to the parent folder "/data/raw_data":

```python
download_path = os.path.join("data", "raw_data")
available_years = ['2016', '2017', '2018', '2019', '2020', '2021']
for year in available_years:
	obj = downloadData(year, download_path)
    obj.download_nhl_raw_data()
```				

The above snippet can be run for any number of times. Note that running the above snippet only download the missing data files and skip the existing data files upon its multiple executions to ensure having all data downloaded and arranged properly using NHL Stats API.

It's worth noticing that since we have two game types (`regular` and `playoff`) for each season, te above code snippet upon execution results in the following data directory structure:

```python
-- /data/raw
    -- 2016
    -- 2017
    -- 2018
    -- 2019
		-- playoff_games
            -- [gameId1].json
            -- [gameId2].json
        -- regular_games
            -- [gameId3].json
            -- [gameId4].json
    -- 2020
	-- 2021
```

### Task 2 - Interactive Debugging Tool

In this task, we delve into NHL data using Interactive Widgets, which enable us to effortlessly switch between Game Types (regular and playoffs) and seasons, as well as view all games. The Interactive Debugging Tool assists us in gaining a deeper understanding of the game data, offering a user-friendly visual representation. For each game, we can sequentially visualize all the event ID's, with events displayed on the ice rink to depict their positions. Additionally, detailed information about the players involved and the specific shot taken. Below is the output image generated using the interactive tool.

![ice_rink](/assets/images/MileStone1.png)

Code written for this tool is given below:
```python
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import ipywidgets as widgets
import json, os

def interactive_tool(game_type):
    """
    Intractive tool main function
    :param game_type: Dropdown to select either regular or playoff game
    """
    # Path of the directory to collect the data. The data for this functionality coming from the extracted jsons
    path_season = '\\'.join(str(os.getcwd()).split('\\')[:-2])
    path_season = os.path.join(path_season, 'Milestone1')
    file_path_season = os.path.join(path_season, 'data', 'raw_data')
    file_path_season = os.path.join(file_path_season, game_type)
    # load the rink image
    rink_path = os.path.join('\\'.join(str(os.getcwd()).split('\\')[:-1]), 'figures', "nhl_rink.png")
    # List all the seasons
    all_files = os.listdir(file_path_season)

    def get_season_data(input_season):
        """
        Function to get the game season data
        :param input_season: Dropdown to select season of the game
        """
        # Path for the data
        path_season = '\\'.join(str(os.getcwd()).split('\\')[:-2])
        path_season = os.path.join(path_season, 'Milestone1')
        file_path_season = os.path.join(path_season, 'data', 'raw_data')
        file_path_season = os.path.join(file_path_season, game_type, input_season + '\\')
        # creating list of all game IDs
        id_list =[filename[6:10] for filename in os.listdir(file_path_season)]

        def get_game_type_id(game_type_id):
            """
            Function to get the ID of the game type
            :param game_type_id: Dropdown to select the game type
            """
            game_type_id = str(game_type_id).zfill(4)
            for filename in os.listdir(file_path_season):
                if filename.endswith(game_type_id + ".json"):
                    f_path = file_path_season + filename
            with open(f_path, 'r') as f_path:
                game_data = json.load(f_path)

            # Get the associated columns of data from json
            print(game_data['gameData']['datetime']['dateTime'])
            game_home = game_data['gameData']['teams']['home']['abbreviation']
            game_away = game_data['gameData']['teams']['away']['abbreviation']
            game_goals_home = game_data['liveData']['plays']['currentPlay']['about']['goals']['home']
            game_goals_away = game_data['liveData']['plays']['currentPlay']['about']['goals']['away']
            game_sog_home = game_data['liveData']['linescore']['periods'][0]['home']['shotsOnGoal'] + game_data['liveData']['linescore']\['periods'][1]['home']['shotsOnGoal'] + game_data['liveData']['linescore']['periods'][2]['home']['shotsOnGoal']
            game_sog_away = game_data['liveData']['linescore']['periods'][0]['away']['shotsOnGoal'] + game_data['liveData']['linescore']\['periods'][1]['away']['shotsOnGoal'] + game_data['liveData']['linescore']['periods'][2]['away']['shotsOnGoal']
            game_so_goalshome = game_data['liveData']['linescore']['shootoutInfo']['home']['scores']
            game_so_attemptshome = game_data['liveData']['linescore']['shootoutInfo']['home']['attempts']
            game_so_goalsaway = game_data['liveData']['linescore']['shootoutInfo']['away']['scores']
            game_so_attemptsaway = game_data['liveData']['linescore']['shootoutInfo']['away']['attempts']
            print("Game ID: "+ game_type_id +";   " + game_home +"  (home) vs "+ game_away + " (away)")
            print ( "           " + str("OT"))
            print("                        " + "Home" +"      "+ "Away" )
            print("           Teams:       " + game_home +"       "+ game_away)
            print("           Goals:       " + f'{game_goals_home}' +"         "+ f'{game_goals_away}')
            print("           SoG:         " + f'{game_sog_home}' +"        "+ f'{game_sog_away}')
            print("           So Goals:    " + f'{game_so_goalshome}' +  "         "+ f'{game_so_goalsaway}')
            print("           So Attempts: " + f'{game_so_attemptshome}' +  "         "+ f'{game_so_attemptsaway}')

            # Store the data into the variable
            game_data = game_data['liveData']['plays']['allPlays']

            def rink_image(img):
                cor = game_data[img]['coordinates']
                if cor=={}:
                    for key, value in game_data[img].items() :
                        print(key, ' : ', value)
                else:
                    plt.imshow(mpimg.imread(rink_path), extent=[-100., 100. ,-42.5, 42.5])
                    title = game_data[img]["result"]["description"]
                    plt.plot(cor['x'], cor['y'], "o")
                    plt.title(title, fontsize = 10, fontweight ='bold')
                    plt.show()
                    for key, value in game_data[img].items():
                        print(key, ' : ', value)
            # Interactive widget for Event Id
            widgets.interact(rink_image, img = widgets.IntSlider(min=0, max=len(game_data)-1, description="Event ID"))

        # Interactive widget for Game ID
        widgets.interact(get_game_type_id, game_type_id = widgets.Dropdown(
            options=id_list,
            description='Select the Game ID:',
            disabled=False))
    # Interactive widget for Season
    widgets.interact(get_season_data, input_season = widgets.Dropdown(
        options=all_files,
        description='Select the Season:',
        disabled=False))
# List of type
widgets.interact(interactive_tool, game_type = widgets.Dropdown(
    options=['regular_season', 'playoffs'],
    description='Select the Game Type:',
    disabled=False))
```

### Task 5 - Advanced Visualizations: Shot Maps

**5.1 - 4 plot offensive zone plots visualization**

We developed this using a Dash Application. We built the application within Dash and then launched it as an overlay on Render. The Dash application is completely interactive, allowing users to make selections based on seasons and teams. You can access our Dash application via the following link:

[Advanced Visualization Shot Map Plot](https://nhlhockeyapp.onrender.com/)

Furthermore, we are providing our HTML plot for your reference:
<iframe src="https://nhlhockeyapp.onrender.com/" title="Advanced Visualizations - Shot Maps" width="990" height="620"> </iframe>

Here is a concise summary of the logs when we deployed our application on Render:

![Deployment Logs](/assets/images/Deploy_logs.png)

**5.2 - Plot interpretation**

Charts with the net positioned at the top illustrate offensive performance, indicating the number of shots taken by the chosen team. This visual representation conveys how the shot frequency compares to the NHL's average team during the same season, focusing on a specific location. The chart employs three distinct colors: 'red,' 'blue,' and 'white,' each representing different scenarios:

The 'red' region indicates that the team has taken shots at a rate higher than the league average.
The 'blue' region signifies that the team has taken shots at a rate lower than the league average.
The 'white' region represents the team's shots taken at the league-average rate.

By analyzing these graphs, one can assess the offensive capabilities of the selected team. The darker the 'red' region, the higher the density of shots for the selected team, while a darker 'blue' region implies a lower density of shots. The 'white' region shows the number of shots taken by the selected team at the league's average rate.

**5.3 - Discussion on Performance Difference of Colorado Avalanche**

In the 2016-17 season, the shot map indicates that the Colorado Avalanche's performance was fairly average. Most of the regions on the rink show that their shot frequency was in line with the league's average rate. Notable exceptions include two blue patterns, signifying that the team scored fewer goals than the league average rate, particularly close to the goal post. There is also a single red region on the left side of the rink, spanning 40 to 50 feet, where the team exceeded the league's average shot rate.

![Colorado Avalanche1](/assets/images/2016_17_Colorado_Avalanche_Team.png)

Fast forward to the 2020-21 season, and the overall shot map highlights a stellar performance by the Colorado Avalanche. The majority of the rink is colored in red, indicating that the team's shot frequency surpassed the league average rate, especially near the goal post, the center of the semi-rink, and the left side between 50 to 60 feet. There are only a few instances where the team's shot frequency was below the league average rate, mainly on the right side of the rink between 50 to 60 feet.

![Colorado Avalanche2](/assets/images/2020_21_Colorado_Avalanche_Team.png)

In summary, the Avalanche team's performance in the 2020-21 season was notably superior, with a larger portion of the rink showing shots exceeding the league average rate compared to the 2016-17 season, where their performance wasn't as strong. This assessment is supported by the prominent presence of red regions on the rink in 2020-21, which was not the case in 2016-17.

**5.4 - Performance Comparison between Buffalo Sabres and Tampa Bay Lightning**

Analyzing the graphs reveals that the Tampa Bay Lightning consistently adheres to a strategy of taking the maximum number of shots in front of the goal post on the rink. In all three seasons, they exhibit a red region, indicating that their shot frequency exceeds the league's average rate, prominently positioned in front of the goal post. This stands in contrast to the Buffalo Sabres, as their red regions consistently appear on the sides of the rink.

Furthermore, the Buffalo Sabres display a darker section within the blue region, with this blue region mostly concentrated on the rink, especially during the 2019-20 and 2020-21 seasons. This pattern contributes to their lackluster performance. Notably, in the 2018-19 season, the Sabres do exhibit a red region, but it is not located in front of the goal post, in contrast to the Tampa Bay Lightning, which underscores another aspect contributing to their underperformance.

In summary, one of the key factors contributing to the Tampa Bay Lightning's success is their consistent ability to take shots in front of the goal post that surpass the league's average rate.

Graphs depicting the performance of the Tampa Bay Lightning in the seasons 2018-19, 2019-20, and 2020-21:-

![Tampa Bay Lightning1](/assets/images/2018_19_Tampa_Bay_Lightning.png)
![Tampa Bay Lightning2](/assets/images/2019_20_Tampa_Bay_Ligtning.png)
![Tampa Bay Lightning3](/assets/images/2020_21_Tampa_Bay_Lightning.png)

Plots illustrating the performance of the Buffalo Sabres in the 2018-19, 2019-20, and 2020-21 seasons:-

![Buffalo Sabres1](/assets/images/2018_19_Buffalo_Sabres_Team.png)
![Buffalo Sabres2](/assets/images/2019_20_Buffalo_Sabres_Team.png)
![Buffalo Sabres3](/assets/images/2020_21_Buffalo_Sabres_Team.png)