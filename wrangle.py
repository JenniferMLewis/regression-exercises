import pandas as pd
import numpy as np


def wrangle_grades():
    '''
    Read student_grades csv file into a pandas DataFrame,
    drop student_id column, replace whitespaces with NaN values,
    drop any rows with Null values, convert all columns to int64,
    return cleaned student grades DataFrame.
    '''
    # Acquire data from csv file.
    df = pd.read_csv('student_grades.csv')
    # Replace white space values with NaN values.
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # Drop all rows with NaN values.
    df = df.dropna()
    # Convert all columns to int64 data types.
    df = df.astype(int)
    return df