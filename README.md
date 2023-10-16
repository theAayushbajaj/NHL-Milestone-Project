# NHL-Milestone-Project

## Overview
- We use modular methods defined in `processed_data.py` to extract raw data from the [NHL API](https://gitlab.com/dword4/nhlapi)
- We generate an interactive plot that helps to visualize events in a given game
- We clean the raw data in JSON format and store it in a more easily workable format - dataframes
- We use the clean data to analyse aggregate and play-by-play data and create interactive visualizations.

# File Structure and Order of Execution
To reproduce the project, run the following notebooks in the given order:
- `src/processed_data.py` (Acquires data from NHL API and stores them in local)
- `src/widget.py` (Generates an interactive tool to visualize events in a game)
- `src/tidyData.py` (Cleans the data and creates a consolidated dataframe of the games)
- `src/Notebooks/02_simple_visualizations.ipynb` (Creates simple visualizations to analyse aggregate data)
- `src/Notebooks/03_advanced_visualizations.ipynb` (Creates advanced interactive visualizations using Plotly)


Given a season, we are interested to compute (1) average shot rate per hour of entire league and (2) excess shot rate per hour for a team. 
This data is provided in four json files in this link: https://drive.google.com/drive/folders/1Z7Tp46Sa4nBaLjChgZVGpMmSeyVzEk7M?usp=sharing

1. excess_20162017
2. excess_20172018
3. excess_20182019
4. excess_20192020

Files with the name excess_<season> contain the average excess shot rate per hour for all seasons, and have the following pattern: <team a>:<excess shot rate per hour for team a>. where each excess shot rate per hour for each team is a numpy array. The x and y coordinates for ploting are also saved as numpy arrays. 

## Advanced Visualization Notes:
### Analysis Assumptions 
There are three types of `periodType`s in the dataset: `SHOOTOUT`, `OVERTIME`, and `REGULAR`. For the sake of considering the each game only lasts 60 minutes, we remove the `events` that happend within `SHOOTOUR` and `OVERTIME`. 

An important factor to consider is the at each round teams change their rink side. For a fair analysis, we map all the `x_coordinates` and `y_coordinates` of the shots into one location (offensive zone). The following function is used to modify the coordinates. Another important fact is that we don't want to consider shots that happened behind the net, and since the net is located 11 ft from the boarder, we remove the shots from this location. 

In addition, for the sake of visualization, we are going to show the distance from the net as the x axis. The frame of which the rink is usually considered has to be modified too. The usuall rink is $[-100, 100]$ but we transform it to $[0, 89]$. 

```python
def XYCoordinateModification(df): 
    # flipp all rows in which the x_coordinate is negative 
    df_copy = df.copy() 
    condition = df['x_coordinate'] < 0 
    df.loc[condition, ['x_coordinate', 'y_coordinate']] *= -1

    # remove those shots behind the net: shots with x_coordinate bigger than (100-11)=89 
    behind_net = df['x_coordinate'] > 89 
    df = df.drop(df[behind_net].index) 
    
    # in the originial rink: the net is place at 89 and the center is placed at 0. 
    # but for visualization according to the milestone, we want to plot the rink such that the net is placed at 0 and the center at 89. 
    # therefore we shift all x_coordinates 89 steps back 
    df['x_coordinate'] = abs(df['x_coordinate'] - 89)
    return df
```


### How average shot map is computed? 
Suppose the games dataframe `dfa` and a season `s` are given. We group the dataframe by the seasons, and only consider a subset of the dataframe involving game events for that season. 

Given the dataframe for the season as `ds`, we group by the `x_coordinate` and `y_coordinate`, and count the totall created groups. We also count how many games have been played in the season `s` by counting the unique `game_id`s in the dataframe `ds`. The average shot rate per hour is calculated as the counts for each group of `x_coordinate` and `y_coordinate` divided by the total games. To make it fair, we divide the final result by 2 to account for the fact that there are two teams in the game. The function `ShotRateGroup(ds)` does this job for us after being applied to the subset of the dataframe. 

For computing the average shot rate per hour for each team in `ds`, we group by the `team_names` in `ds`, and perform the previous process. We don't need to divide the final result by 2, as we are only considering one team. Then we subtract the result from the average shot rate per hour of that season. See this example: 

```python
Excess = lambda team_rate, league_rate: team_rate.sub(league_rate, fill_value = 0)
season = 20172018
shot_all = all_season_shot_rate[season]
dfb = dfa[dfa['season'] == season] 
team_names = dfb['team_name'].unique() 
team_shots = dfb.groupby('team_name').apply(ShotRateGroup)
excess  = {name: Excess(team_shots.loc[name], shot_all) for name in team_names}
excess_smooth = {name:Smoother(excess_20162107[name]) for name in team_names}
```

The function `Smoother` is only for a better visualizations performance. This function uses kernel density estimation to smooth out the sharp edges of the contour plot. 

Here is the final result: 


<img src="data/excess_shot_rate_example.png" alt="example" width="1000"/>

### How a single figure is plotted? 
Given the `season = 20162017` and `team_name = 'Toronto Maple Leafs'`, the average shot rate for the entire league is computed from `ShotRate(df)` and then `ExcessRate(team, season, df_rate)` function computed the excess shot rate per hour. The result is smoothen out by `SmoothOut(team_name, season, dfa)`. The function `PlotShotMap(team_name, season, z_smooth, x, y)` creates the figure.  

