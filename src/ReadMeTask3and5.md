# Baseline Model 
The program is called `baseline.py`. For this purpose, we train a simple logistic regression model on different sets of features from the First Feature Engineering Task. 
1. `baseline.py`: This part trains and validates a simple logisitc regression model using the features from the first feature engineering task. In addition, experiments are tracked. 
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
5. `task5_q3_part_1`: In this part, we use feature selection methods from sklearn to identify the optimal set of features.
6. `task5_q3_part_2`: In this part, we use shap library to identify the optimal set of features.
7. `task5_q3_part_3`: In this part, we tune the hyperparameters with the found optimal set of features. 


# Links for Experiment Tracking: 
1. [Task 3](https://www.comet.com/2nd-milestone/baseline-model/d485ba3099ca4d9694823b2bf5ae0721?experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=wall)
2. [Task 5-q1](https://www.comet.com/2nd-milestone/baseline-model/15cc16e53b304c8c83e9c015dc812ebf](https://www.comet.com/2nd-milestone/advanced-model/06018ec9b8f9422292056cc314392b69)
%[Task 5-q1-tuned](https://www.comet.com/2nd-milestone/baseline-model/b318d8d8e1e048189627095217d6865a)%
6. [Task 5-q2](https://www.comet.com/2nd-milestone/baseline-model/02098092281a4629b09c8a6b6e04ee4a](https://www.comet.com/2nd-milestone/advanced-model/3df4529c3a454917bd976ba2f322a9ef)
7. [Task 5-q2-tuned](https://www.comet.com/2nd-milestone/baseline-model/89d296cacbde4c38b1e2ceed6763eaa2](https://www.comet.com/2nd-milestone/baseline-model/f3324ca1d476444d994c28ec555b5133)
8. [Task 5-q3-tuned](https://www.comet.com/2nd-milestone/baseline-model/4a6aa628377e48a2a71af99bce75c14b?experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step) 
