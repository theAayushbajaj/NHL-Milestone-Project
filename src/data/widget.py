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
            game_sog_home = game_data['liveData']['linescore']['periods'][0]['home']['shotsOnGoal'] + game_data['liveData']['linescore']\
            ['periods'][1]['home']['shotsOnGoal'] + game_data['liveData']['linescore']['periods'][2]['home']['shotsOnGoal']
            game_sog_away = game_data['liveData']['linescore']['periods'][0]['away']['shotsOnGoal'] + game_data['liveData']['linescore']\
            ['periods'][1]['away']['shotsOnGoal'] + game_data['liveData']['linescore']['periods'][2]['away']['shotsOnGoal']
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