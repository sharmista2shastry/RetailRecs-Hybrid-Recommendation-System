import numpy as np
import pandas as pd

def unique_users(data, column):
    """
    Returns a sorted list of unique user IDs
    """
    return np.sort(data[column].unique())

def unique_items(data, column):
    """
    Returns a list of unique Product Names
    """
    return data[column].unique()

def features_to_add(data, column1, column2, column3):
    data1 = data[column1]
    data2 = data[column2]
    data3 = data[column3]

    return pd.concat([data1, data2, data3], ignore_index=True).unique()

def mapping(users, items, features):
    user_to_index_mapping = {}
    index_to_user_mapping = {}

    for user_index, user_id in enumerate(users):
        user_to_index_mapping[user_id] = user_index
        index_to_user_mapping[user_index] = user_id
    
    item_to_index_mapping = {}  
    index_to_item_mapping = {}  
    for item_index, item_id in enumerate(items):
        item_to_index_mapping[item_id] = item_index
        index_to_item_mapping[item_index] = item_id
        
    feature_to_index_mapping = {}
    index_to_feature_mapping = {}
    for feature_index, feature_id in enumerate(features):
        feature_to_index_mapping[feature_id] = feature_index
        index_to_feature_mapping[feature_index] = feature_id
    
    return user_to_index_mapping, index_to_user_mapping, item_to_index_mapping, index_to_item_mapping, feature_to_index_mapping, index_to_feature_mapping