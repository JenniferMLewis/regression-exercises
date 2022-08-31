import pandas as pd
import numpy as np
import os
from env import user, password, host


def get_url(db, user=user, password=password, host=host):
    '''
    take database name for input,
    returns url, using user, password, and host pulled from your .env file.
    PLEASE save it as a variable, and do NOT just print your credientials to your document.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


def get_zillow():
    '''
    Returns the zillow 2017 dataset, checks local disk for zillow_2017.csv, if present loads it,
    otherwise it pulls the bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
    taxvaluedollarcnt, yearbuilt, taxamount and, fips columns from the SQL.
    '''
    filename = 'zillow_2017.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        df = pd.read_sql('''SELECT bedroomcnt, 
        bathroomcnt, 
        calculatedfinishedsquarefeet, 
        taxvaluedollarcnt, 
        yearbuilt, 
        taxamount, 
        fips 
        FROM properties_2017
        JOIN propertylandusetype
        USING(propertylandusetypeid)
        WHERE propertylandusedesc = "Single Family Residential";
        ''', get_url('zillow'))
        df.to_csv(filename)
        return df


def get_zillow_inferred():
    '''
    Returns the zillow 2017 dataset, checks local disk for zillow_2017.csv, if present loads it,
    otherwise it pulls the bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, 
    taxvaluedollarcnt, yearbuilt, taxamount and, fips columns for Single Family Residential Property types from the SQL.
    This varies from get_zillow by also pulling Inferred Single Family Residential Property types.
    '''
    filename = 'zillow_2017_inferred.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        df = pd.read_sql('''SELECT bedroomcnt, 
        bathroomcnt, 
        calculatedfinishedsquarefeet, 
        taxvaluedollarcnt, 
        yearbuilt, 
        taxamount, 
        fips 
        FROM properties_2017
        JOIN propertylandusetype
        USING(propertylandusetypeid)
        WHERE propertylandusedesc = "Single Family Residential", "Inferred Single Family Residential";
        ''', get_url('zillow'))
        df.to_csv(filename)
        return df


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


    def remove_outliers(df, k, col_list):
        ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
        '''
    
        for col in col_list:

            q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
            iqr = q3 - q1   # calculate interquartile range
        
            upper_bound = q3 + k * iqr   # get upper bound
            lower_bound = q1 - k * iqr   # get lower bound

            # return dataframe without outliers
        
            df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
        return df