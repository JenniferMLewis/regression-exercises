import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer

def scale_data(train, 
               validate, 
               test, 
               scaler,
               columns_to_scale=['bedrooms', 'bathrooms', 'tax_value']):
    '''
    Scales a copy of train, validate, and test dataset. Please, feed me a scaler, Seymour.
    Takes train, validate, and test data, then splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     fit the thing
    scaler.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(validate[columns_to_scale]),
                                                  columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(scaler.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    return train_scaled, validate_scaled, test_scaled

def visualize_scaler(scaler,
                     df,
                     columns_to_scale=['bedrooms', 'bathrooms', 'tax_value'],
                     bins=10):
    '''
    Visualises data using the provided Scaler.
    Takes in Scaler, Data Frame, columns being scaled (defaults to 'bedbrooms', 'bathrooms' and 'tax_value),
    and bin the data for visualisation (defaults to 10 bins).
    Returns 
    '''
    fig, axs = plt.subplots(len(columns_to_scale), 2, figsize=(16,9))
    df_scaled = df.copy()
    df_scaled[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    for (ax1, ax2), col in zip(axs, columns_to_scale):
        ax1.hist(df[col], bins=bins)
        ax1.set(title=f'{col} before scaling', xlabel=col, ylabel='count')
        ax2.hist(df_scaled[col], bins=bins)
        ax2.set(title=f'{col} after scaling with {scaler.__class__.__name__}', xlabel=col, ylabel='count')
    plt.tight_layout()