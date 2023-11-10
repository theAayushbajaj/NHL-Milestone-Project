# Baseline Model 
The program is called `baseline.py`. For this purpose, we train a simple logistic regression model on different sets of features from the First Feature Engineering Task. 
1. `baseline.py`: This part trains and validates a simple logisitc regression model using the features from the first feature engineering task. 
2. `baseline2.py`: Creates all figures into one.
3. `utils_ar_pa.py`: Helper functions for creating figures and formatings. Student Paul wrote the part for figure creation. 

# Advanced Model 
This task is done in multiple parts. 
1. `task5_q1_part_1`: This part Standardizes the columns with different features, and trains an xgboost model on features of the first feature engineering task, without any hyperparameter tuning. 
2. `task5_q2_part_2`: This part tunes the hyperparameters of the above model. 
1. `task5_q2_part_1`: This part deals with `Nans` in the Feature Engineering Task.
2. `task5_q2_part_2`: This part Encodes the Categorical Columns and Standardizes the columns with different features. 
3. `task5_q2_part_3`: This part trains and validates an xgboost model without hyperparameter tuning. 
4. `task5_q2_part_4`: This part finds the best xgboost model with gridsearch cross-validation and validates the best model.

