import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

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
        df = pd.read_csv(filename)
        df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                      'bathroomcnt':'bathrooms', 
                      'calculatedfinishedsquarefeet':'area',
                      'taxvaluedollarcnt':'tax_value', 
                      'yearbuilt':'year_built',
                      'taxamount':'tax_amount'})
        df = df.drop(columns="Unnamed: 0")
        df['bedrooms'] = df.bedrooms.astype(float)
        df['year_built'] = df.year_built.astype(str)
        return df
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
        df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                      'bathroomcnt':'bathrooms', 
                      'calculatedfinishedsquarefeet':'area',
                      'taxvaluedollarcnt':'tax_value', 
                      'yearbuilt':'year_built',
                      'taxamount':'tax_amount'})
        df = df.drop(columns="Unnamed: 0")
        df['bedrooms'] = df.bedrooms.astype(float)
        df['year_built'] = df.year_built.astype(str)
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
    
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                          'bathroomcnt':'bathrooms', 
                          'calculatedfinishedsquarefeet':'area',
                          'taxvaluedollarcnt':'tax_value', 
                          'yearbuilt':'year_built',
                          'taxamount':'tax_amount'})
    df = df.drop(columns="Unnamed: 0")
    
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
    ''' 
    remove outliers from a list of columns in a dataframe 
    and return that dataframe.

    '''
    for col in col_list:
        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        iqr = q3 - q1   # calculate interquartile range

        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound
        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df


def hist_plot(df):
    '''
    Plots Histograms for columns in input df, all but 'fips' and, 'year_built' as they're categorical not a
    ctually numbers.
    '''
    plt.figure(figsize=(16, 3))

    cols = [col for col in df.columns if col not in ['fips', 'year_built']]

    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1 <-- Good to note
        plot_number = i + 1 
        plt.subplot(1, len(cols), plot_number)
        plt.title(col)
        df[col].hist(bins=5)
        # We're looking for shape not actual details, so these two are set to 'off'
        plt.grid(False)
        plt.ticklabel_format(useOffset=False)
        # mitigate overlap: This is handy. Thank you.
        plt.tight_layout()

    plt.show()

def box_plot(df):
    ''' Plots Boxplots of bedrooms, bathrooms, area, tax_value, and tax_amount'''
    
    # List of columns
    cols = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'tax_amount']

    plt.figure(figsize=(16, 3))

    for i, col in enumerate(cols):
        plot_number = i + 1 
        plt.subplot(1, len(cols), plot_number)
        plt.title(col)
        sns.boxplot(data=df[[col]])
        plt.grid(False)
        plt.tight_layout()

    plt.show()

# -----------------------Full Defs-------------------------

# Having this section with the full cleaning is such a great idea for the project,
# Clean, organise, and everything with as few lines as possible.

def prepare_zillow(df):
    ''' Prepare zillow data for exploration'''

    # removing outliers
    df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'area', 'tax_value', 'tax_amount'])
    
    # getting distributions for numeric data
    hist_plot(df)
    box_plot(df)
    
    # converting column datatypes
    df.fips = df.fips.astype(str)
    df.year_built = df.year_built.astype(str)
    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
    # impute year built using median (If the teacher is willing to, then it -should- be okay?)
    imputer = SimpleImputer(strategy='median')

    imputer.fit(train[['year_built']])

    train[['year_built']] = imputer.transform(train[['year_built']])
    validate[['year_built']] = imputer.transform(validate[['year_built']])
    test[['year_built']] = imputer.transform(test[['year_built']])       
    
    return train, validate, test 


# -------------------------------- One Liner -------------------------

# The paydirt?!
def wrangle_zillow():
    '''Acquire and prepare data from Zillow database for explore using acquire and prepare functions above.'''
    train, validate, test = prepare_zillow(get_zillow())
    
    return train, validate, test