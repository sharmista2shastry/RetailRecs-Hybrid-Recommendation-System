import pandas as pd
from scipy.sparse import coo_matrix

def read_data(data_dir, filename):
    """
    Reads a data file from the specified directory and returns it as a pandas DataFrame.
    
    Parameters:
    data_dir (str): The directory where the data files are stored.
    filename (str): The name of the data file to read.
    
    Returns:
    pd.DataFrame: The data read from the file as a DataFrame.
    """

    try:
        df = pd.read_excel(data_dir, filename)
    except Exception as e:
        print(e)
    else:
        return df

def merge_dataset(df1, df2, left_on_param, right_on_param, join_type):
    try:
        final_df = pd.merge(df1, df2, left_on=left_on_param, right_on=right_on_param, how=join_type)
    except Exception as e:
        print(e)
    else:
        return final_df

def interactions(data, row, col, values, row_map, col_map):
    row = data[row].apply(lambda x: row_map[x]).values
    col = data[col].apply(lambda x: col_map[x]).values
    values = data[values].values

    return coo_matrix((values, (row, col)), shape=(len(row_map), len(col_map)))