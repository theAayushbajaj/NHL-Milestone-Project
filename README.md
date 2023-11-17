# NHL-Milestone-Project

## Overview
- We use modular methods defined in `processed_data.py` to extract raw data from the [NHL API](https://gitlab.com/dword4/nhlapi)
- We generate an interactive plot that helps to visualize events in a given game
- We clean the raw data in JSON format and store it in a more easily workable format - dataframes
- We use the clean data to analyse aggregate and play-by-play data and create interactive visualizations.

## File Structure and Order of Execution
To reproduce the project, run the following files in the given order:
- `src/02_feature_engg.py` 
- `src/03_baseline_models.py` 
- `src/04_feature_engg.py` 
- `src/05_advanced_models.ipynb` 
- `src/06_best_shot_balanced_RF.py`
- `src/06_best_shot_mlp.py` 
- `src/06_best_shot_rfsmote.py` 
- `src/06_best_shot_svmBagging.py` 
- `src/06_best_shot_tfNN.py`
- `src/06_best_shot_VotinggClassifier.py`
- `src/06_best_shot_weighted_logistic.py`
- `src/06_best_shot_xgboost_smote.py`
- `src/07_test_eval.py`
- `src/utils.py` 
