import streamlit as st
import pandas as pd
import os
import numpy as np
from ift6758.client import ServingClient
from ift6758.client import GameClient
# from ift6758.client import ServingClient
# from ift6758.client import GameClient
from sklearn.preprocessing import MinMaxScaler as mscale
import json
from sklearn.linear_model import LogisticRegression as logReg
from sklearn.model_selection import train_test_split
import traceback

# Setting the app title
st.title("Hockey Visualization App")
sc = ServingClient.ServingClient(ip="docker-project_milestone_3-serving-1", port=5000)
# sc = ServingClient.ServingClient(ip="127.0.0.1", port=5000)

final_data = pd.read_csv(os.path.join(os.getcwd(), 'ift6758/ift6758/data/all_data_final_3.csv'))
# data_df = final_data[['game_id', 'shot_distance', 'shot_angle', 'empty_net', 'is_goal']]
x = final_data[['shot_distance', 'shot_angle', 'empty_net']]
y = final_data[['is_goal']]
modell = logReg()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
modell.fit(x_train, y_train)

with st.sidebar:
    # This sidebar uses Serving Client for the model Download
    workspace = st.text_input('Workspace', 'Workspace x')
    model = st.text_input('Model', 'Model y')
    version = st.text_input('Version', 'Version z')
    if st.button('Get Model'):
        sc.download_registry_model(
            workspace=workspace,
            model=model,
            version=version
        )
        st.write('Model Downloaded')
        if 'track_download_model' not in st.session_state and 'previous_track_download_model' not in st.session_state:
            st.session_state['track_download_model'] = 1
            st.session_state['previous_track_download_model'] = 1
        else:
            st.session_state['track_download_model'] += 1


def ping_game_id(game_id):
    with st.container():
        # Keep the game_id in session_state
        if 'game_id' not in st.session_state:
            st.session_state['game_id'] = game_id
        if st.session_state.game_id != game_id and 'track_download_model' not in st.session_state and 'previous_track_download_model' not in st.session_state:
            st.session_state['track_download_model'] = 1
            st.session_state['previous_track_download_model'] = 1
        elif st.session_state.game_id != game_id and 'track_download_model' in st.session_state and 'previous_track_download_model' in st.session_state:
            if st.session_state['track_download_model'] == (st.session_state['previous_track_download_model'] + 1):
                st.session_state.game_id = game_id
                st.session_state['previous_track_download_model'] = st.session_state['track_download_model']
        # Initialization of session variables to track the dataframe length
        # st.session_state preserves the state of the variables between different reruns
        # session_state is the key functionality in streamlit app
        if 'session_tracker' not in st.session_state and 'previous_session_tracker' not in st.session_state:
            st.session_state['session_tracker'] = 0
            st.session_state['previous_session_tracker'] = 0
            # st.write(st.session_state.session_tracker)
            # st.write(st.session_state.previous_session_tracker)
        if st.session_state.game_id != game_id:
            st.session_state['session_tracker'] = 0
            st.session_state['previous_session_tracker'] = 0
            st.write("Model trained on different game id should not predict goal probabilities of different game id. "
                     "Please download the model again and perform prediction for this new game id.")
            # st.write("We have used the 'event type' feature and since this feature does not have all the "
            #          "types used by the previous game id, therefore, you should train your model "
            #          "again with the new events set to get the predictions and avoid Features Mismatch Problem."
            #          "Please stop the application service and retrain your model again on new game id.")
        try:
            if st.session_state['game_id'] == game_id:
                # Get the filepath of the recent game_id downloaded json
                gc = GameClient.GameClient(game_id=game_id)
                new_events = gc.timed_ping_api()

            if st.session_state.session_tracker == st.session_state.previous_session_tracker and st.session_state['game_id'] == game_id:
                st.subheader(
                    "Game " + str(new_events[9]) + ": " + str(new_events[10]) + " vs " + str(new_events[11]))
                st.text("Period " + str(new_events[12]) + " - "
                        + str(new_events[13]) + " left")
                model_df = new_events[16]
                x_model_df = model_df[['shot_distance', 'shot_angle', 'is_emptynet']]
                scalar = mscale()
                test_model_df = scalar.fit_transform(x_model_df)
                predictions = modell.predict_proba(test_model_df)
                max_values = np.max(predictions, axis=1)
                max_values = max_values.reshape(-1, 1)
                col1, col2 = st.columns(2)
                model_df['model_predictions'] = max_values
                grouped_prob_df = model_df.where(model_df.is_goal == 1).groupby("team")["model_predictions"].agg("sum").round(decimals=2)
                grouped_goal_df = model_df.where(model_df.is_goal == 1).groupby("team")["is_goal"].count()
                grouped_prob_df = pd.DataFrame(grouped_prob_df).transpose()
                grouped_goal_df = pd.DataFrame(grouped_goal_df).transpose()
                col1.metric(label=str(grouped_prob_df.columns[0]) + " xG(actual)",
                            value=str(grouped_prob_df[[grouped_prob_df.columns[0]]][grouped_prob_df.columns[0]]['model_predictions']) + " ("
                                  + str(grouped_goal_df[[grouped_goal_df.columns[0]]][grouped_goal_df.columns[0]]['is_goal'])
                                  + ")",
                            delta=str(float(float(grouped_prob_df[[grouped_prob_df.columns[0]]][grouped_prob_df.columns[0]]['model_predictions']) - float(
                                grouped_goal_df[[grouped_goal_df.columns[0]]][grouped_goal_df.columns[0]]['is_goal'])).__round__(2)))
                col2.metric(label=str(grouped_prob_df.columns[1]) + " xG(actual)",
                            value=str(grouped_prob_df[[grouped_prob_df.columns[1]]][grouped_prob_df.columns[1]]['model_predictions']) + " ("
                                  + str(grouped_goal_df[[grouped_goal_df.columns[1]]][grouped_goal_df.columns[1]]['is_goal'])
                                  + ")",
                            delta=str(float(float(grouped_prob_df[[grouped_prob_df.columns[1]]][grouped_prob_df.columns[1]]['model_predictions']) - float(
                                grouped_goal_df[[grouped_goal_df.columns[1]]][grouped_goal_df.columns[1]]['is_goal'])).__round__(2)))

                try:
                    with st.container():
                        # Fetching the entire dataframe
                        st.subheader("Data used for predictions (and predictions)")
                        st.write(model_df)
                except Exception as e:
                    print(e)
                    pass
        except Exception as e:
            st.write("Please turn on your prediction service.")
            st.write(e)
            st.write(traceback.format_exc())
            st.session_state['session_tracker'] = 0
            st.session_state['previous_session_tracker'] = 0


with st.container():
    # This is the ping game container consists of the Game ID and button
    game_id = st.text_input('Game ID', '2021020329')
    if st.button('Ping game'):
        ping_game_id(game_id)
