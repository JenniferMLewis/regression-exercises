import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

alf = 0.05

def dtypes_to_list(df):
    '''
    Takes in a dataframe, returns two lists, 
    one of num type column names, and one of categorical type column names.
    '''
    num_type_list, cat_type_list = [], []
    for column in df:
        col_type =  df[column].dtype
        if col_type == "object" :
            cat_type_list.append(column)
        if np.issubdtype(df[column], np.number) and \
            ((df[column].max() + 1) / df[column].nunique())  == 1 :
            cat_type_list.append(column)
        if np.issubdtype(df[column], np.number) and \
            ((df[column].max() + 1) / df[column].nunique()) != 1 :
            num_type_list.append(column)
    return num_type_list, cat_type_list

def col_range(df):
    '''
    Takes in a data frame, returns the 'describe' of the data frame with a new entry 'range'.
    'Range' is the difference between the 'max' and 'min' columns.
    '''
    stats_df = df.describe().T
    stats_df['range'] = stats_df['max'] - stats_df['min']
    print(stats_df)

def explore_cat(df, cat, target):
    '''
    Takes in dataframe (Remember, explore TRAIN!), 
    categorial columns (please [make, it, a, list]),
    target column, returns printed value counts for each category in each column
    '''
    for col in cat:
        print(col)
        print(df[col].value_counts())
        print(df[col].value_counts(normalize=True)*100)
        sns.countplot(x=col, data=df)
        plt.title(col + ' counts')
        plt.show()
    
        sns.barplot(data=df, x=col, y=target)
        rate = df[target].mean()
        plt.axhline(rate, label= 'average ' + target + ' rate')
        plt.legend()
        plt.title(target + ' rate by ' + col)
        plt.show()
    
        o = pd.crosstab(df[col], df[target])
        chi2, p, dof, e = stats.chi2_contingency(o)
        result = p < alf
        if result == True:
            print('P is less than Alpha.')
        else:
            print("P is greater than Alpha.")
        print('---------------------------------------------')

def explore_num(df, num):
    '''
    Takes in DataFrame (Remember, explore w/ Train!), 
    numerical columns to explore (as a list).
    '''
    for col in num:
        sns.histplot(x=col, data=df)
        plt.show()
        
def plot_chi(df, cat, target):
    '''
    Takes in Dataframe (Use Train, please),
    categorical columns (list),
    and the target column.
    creates a Chi squared for each categorical column vs. target column.
    '''
    for col in cat:
        o = pd.crosstab(df[col], df[target])
        chi2, p, dof, e = stats.chi2_contingency(o)
        plt.plot(df[col], chi2)
    plt.show()    
    
    
def select_kbest(X_train, y_train, k_features):
    '''
    This function takes in X_train, y_train, and the number of features
    to select and returns the names of the selected features using SelectKBest
    from sklearn. 
    '''
    kbest = SelectKBest(f_regression, k=k_features)
    kbest.fit(X_train, y_train)
    
    print(X_train.columns[kbest.get_support()].tolist())
    
    
def select_rfe(X_train, y_train, k_features):
    '''
    This function takes in X_train, y_train, and the number of features
    to select and returns the names of the selected features using Recursive
    Feature Elimination from sklearn. 
    '''
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=k_features)
    rfe.fit(X_train, y_train)
    
    print(X_train.columns[rfe.support_].tolist())
    
    
def select_sfs(X_train, y_train, k_features):
    '''
    This function takes in X_train, y_train, and the number of features
    to select and returns the names of the selected features using Sequential
    Feature Selector from sklearn. 
    '''
    model = LinearRegression()
    sfs = SequentialFeatureSelector(model, n_features_to_select=k_features)
    sfs.fit(X_train, y_train)
    
    print(X_train.columns[sfs.support_].tolist())