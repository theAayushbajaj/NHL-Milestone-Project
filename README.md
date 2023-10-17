# NHL-Milestone-Project

## Overview
- We use modular methods defined in `processed_data.py` to extract raw data from the [NHL API](https://gitlab.com/dword4/nhlapi)
- We generate an interactive plot that helps to visualize events in a given game
- We clean the raw data in JSON format and store it in a more easily workable format - dataframes
- We use the clean data to analyse aggregate and play-by-play data and create interactive visualizations.

## File Structure and Order of Execution
To reproduce the project, run the following notebooks in the given order:
- `src/processed_data.py` (Acquires data from NHL API and stores them in local)
- `src/widget.py` (Generates an interactive tool to visualize events in a game)
- `src/tidyData.py` (Cleans the data and creates a consolidated dataframe of the games)
- `src/Notebooks/02_simple_visualizations.ipynb` (Creates simple visualizations to analyse aggregate data)
- `src/Notebooks/03_advanced_visualizations.ipynb` (Creates advanced interactive visualizations using Plotly)
