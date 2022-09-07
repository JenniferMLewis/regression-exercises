import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

alf = 0.05
cat = ['bedrooms', 'bathrooms',  'year_built',  'fips']
num = ['area', 'tax_amount', 'tax_value']

### This still needs work, it's been returning bedroom and fips for cat, and not bath/year built as well.
# Will have to manually list out cat and num data.

# def dtypes_to_list(df):
#     '''
#     Takes in a dataframe, returns two lists, 
#     one of num type column names, and one of categorical type column names.
#     '''
#     num_type_list, cat_type_list = [], []
#     for column in df:
#         col_type =  df[column].dtype
#         if col_type == "object" :
#             cat_type_list.append(column)
#         elif np.issubdtype(df[column], np.number) and \
#             ((df[column].max() + 1) / df[column].nunique())  == 1 :
#             cat_type_list.append(column)
#         elif np.issubdtype(df[column], np.number) and \
#             ((df[column].max() + 1) / df[column].nunique()) != 1 :
#             num_type_list.append(column)
#     return num_type_list, cat_type_list

def col_range(df):
    '''
    Takes in a data frame, returns the 'describe' of the data frame with a new entry 'range'.
    'Range' is the difference between the 'max' and 'min' columns.
    '''
    stats_df = df.describe().T
    stats_df['range'] = stats_df['max'] - stats_df['min']
    print(stats_df)

def plot_variable_pairs(df, num = ['area', 'tax_amount', 'tax_value']):
    '''
    Takes in DF (Train Please,) and plots out the variable pairs heatmap and pairplot. 
    Preset Categorical data is bedrooms, bathrooms, year_built, and fips, but a different list can be fed in.
    Preset Numerical Data is area, tax_amount, tax_value, but a new list can be fed in.
    '''
    df_corr = df.corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(df_corr, cmap='Purples', annot = True, mask= np.triu(df_corr), linewidth=.5)
    plt.show()
    
    sns.pairplot(df[num].sample(1_000), corner=True, kind='reg', plot_kws={'line_kws':{'color':'red'}})
    plt.show()

def plot_categorical_and_continuous_vars(df, cat = ["bedrooms", "bathrooms"], target = 'tax_value', hues = "fips"):
    '''
    plots a continuous varible (please enter an int/float column) as y
    sorted by categorical variable [default bedrooms, bathrooms] (Year_built not included, there's over 100 years, not exactly categorical to the scale we want.) as x
    and hue based upon 'fips' [this can be changed, just hues = "whatever_here", I just want to see colour by fips, "none" to make it stop].
    returns swarm plot, violin plot, and cat plot.
    '''
    for col in cat:
        sns.swarmplot(data=df.sample(800), x=col, y=target, hue=hues, s=3)
        plt.show()

        sns.violinplot(data=df.sample(1_000), x=col, y=target, hue=hues, s=3)
        plt.show()
        
        sns.catplot(data=df.sample(500), x=col, y=target, hue=hues, s=2)
        plt.show()

def explore_cat(df, cat = ['bedrooms', 'bathrooms',  'year_built',  'fips'], target = 'tax_value'):
    '''
    Takes in dataframe (Remember, explore TRAIN!), 
    categorial columns (default is bedrooms, bathrooms, year_built, and fips),
    target column, returns printed value counts for each category in each column
    '''
    for col in cat:
        print(f"*** {col} ***")
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

def explore_num(df, num = ['area', 'tax_amount', 'tax_value']):
    '''
    Takes in DataFrame (Remember, explore w/ Train!), 
    numerical columns to explore (default area, tax_amount, tax_value).
    '''
    for col in num:
        sns.histplot(x=col, data=df)
        plt.show()  

        sns.scatterplot(data=df, x='col', y='tax_value')
        plt.show()
    
    
# ------------- Feature Selection -------------------

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

