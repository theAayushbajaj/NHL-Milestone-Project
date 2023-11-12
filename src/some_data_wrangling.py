#%%
# Number of missing values per column in X_train
data_fe2.isnull().sum()
#%%

# bool indexing speed is na
tmp = data_fe2[data_fe2['speed'].isna()]

#%%

tmp['is goal'].value_counts()
# %%
