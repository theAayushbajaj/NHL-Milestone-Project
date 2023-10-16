from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import numpy as np
import pandas as pd
from PIL import Image

app = Dash(__name__)
server = app.server
nhl_df = pd.read_csv('complex_diff.csv')

seasons = {}
for season in nhl_df.season.unique().tolist():
    season_df = nhl_df.loc[nhl_df['season'] == season]
    seasons[season] = list(set(season_df['team shot'].unique().tolist()))

app.layout = html.Div([
    html.Div([
        dcc.RadioItems(
            list(seasons.keys()), list(seasons.keys())[0],
            id='season-radio', inline=True,
        ),
        html.Hr(), dcc.RadioItems(id='team-radio', inline=True),
        html.Hr(), html.Div(id='display-selected-values')
    ]),
    dcc.Graph(id='nhl-graph-data')
], 
style = {
    'display': 'inline-block', 'vertical-align': 'top'
})

@app.callback(
    Output('team-radio', 'options'),
    Input('season-radio', 'value'))
def select_season(selected_season):
    return [{'label': i, 'value': i} for i in seasons[selected_season]]

@app.callback(
    Output('team-radio', 'value'),
    Input('team-radio', 'options'))
def team_data(available_options):
    return available_options[0]['value']

@app.callback(
    Output('display-selected-values', 'children'),
    Input('season-radio', 'value'),
    Input('team-radio', 'value'))
def show_info(selected_season, selected_team):
    return u'{} is a team played in {} season'.format(
        selected_team, selected_season,
    )

@app.callback(
    Output('nhl-graph-data', 'figure'),
    Input('season-radio', 'value'),
    Input('team-radio', 'value'))
def modify_plot(selected_season, selected_team):
    im = Image.open('nhl_rink.png')
    rot_img = im.transpose(Image.Transpose.ROTATE_90)
    img_width, img_height = rot_img.size
    nhl_team = nhl_df.loc[(nhl_df['team shot'] == selected_team) & (nhl_df['season'] == selected_season)]
    x_rink, y_rink = np.sort(nhl_team['y_mid'].unique()), np.sort(nhl_team['goal_mid'].unique())
    [x, y] = np.meshgrid(x_rink, y_rink)
    difference = griddata((nhl_team['y_mid'], nhl_team['goal_mid']), nhl_team['raw_diff'], (x, y), method='cubic', fill_value=0)
    difference = gaussian_filter(difference, sigma=1.5)
    min_difference, max_difference = np.min(difference), np.max(difference)
    if np.abs(min_difference) > np.abs(max_difference): max_difference = np.abs(min_difference)
    else: min_difference = -np.abs(max_difference)

    nhl_figure = go.Figure(data=
    go.Contour(
        z=difference, x=x_rink, y=y_rink, opacity=0.7, zmin=min_difference,
        zmax=max_difference, colorscale=[[0, '#0000FF'], [0.5, 'white'], [1, '#FF0000']],
    ))
    nhl_figure.update_yaxes(autorange="reversed", showgrid=False)
    nhl_figure.add_layout_image(
        dict(
            source=rot_img, xref="x", yref="y", x=-40, y=-10, sizex=img_width/6, sizey=img_height/5.5,
            sizing="stretch", opacity=1, layer="below"
            )
    )
    nhl_figure.add_trace(
        go.Scatter(
            x=[-40, 40], y=[-4, None, 90], showlegend=False
            )
    )
    nhl_figure.update_xaxes(range=[-40, 40], title_text='Distance from the center of the rink(ft)', showgrid=False)
    nhl_figure.update_yaxes(range=[-10, 90], title_text='Distance from the goal line(ft)', showgrid=False)
    nhl_figure.update_yaxes(tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90], showgrid=False)
    nhl_figure.update_layout(
        autosize=False, width=500, height=500, template="plotly_white", title="5v5 Offence"
    )
    return nhl_figure


if __name__ == '__main__':
    app.run_server(debug=True)
