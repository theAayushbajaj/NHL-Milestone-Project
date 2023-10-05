import numpy as np
import pandas as pd


#%%

# get the proportion

def add_prop(group):
    group['prop'] = group.count()/group.count().sum()
    return group
    

#%%

def euclidian_distance_goal(x_shot : int, y_shot : int,
                            period : int , home = True) -> float:
    """
    

    Parameters
    ----------
    x_shot : int
        DESCRIPTION.
    y_shot : int
        DESCRIPTION.
    period : int
        DESCRIPTION.
    home : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    float
        DESCRIPTION.

    """
    y_goal = 0
    
    # if home:
    #     x_goal = 89 if period%2 == 1 else -89
    # else:
    #     x_goal = -89 if period%2 == 0 else 89
        
    x_goal = 89 if x_shot > 0 else -89
        
        
    return np.linalg.norm(np.array([x_shot, y_shot]) - np.array([x_goal, y_goal]))
    
