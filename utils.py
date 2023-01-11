# dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def get_data(url='https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv',
             transform_series=True,
             target_col='CPIAUCSL',
             target_col_transformation=8):
    '''
    This function loads the data

    :param url: link to FREDMD macro data set 'https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv'
    :param transform_series: transform series to make them stationary following McCracken and Ng (2015) (except CPI)
    :param target_col: which column contains the target
    :param target_col_transformation: which transformation for target. Default is 8 for CPI, i.e. YoY-change
    :return: either raw or transformed dataframe and oos_sample data
    '''

    # read csv and set index
    df = pd.read_csv(url, index_col='sasdate')

    # store transformations
    transformations = df.iloc[0]

    # change transformation for target (if different transformation wished than McCracken and Ng (2015))
    transformations[target_col] = target_col_transformation

    # remove first row which contains transformations
    df = df.iloc[1:]

    # change index to datetime
    df.index = pd.to_datetime(df.index)


    # transform series
    if transform_series:
        for col in df.columns:
            df[col] = _transform_series(df[col], transformations[col])

        # if YoY-change, inflation rows will contain NaN values until 1960-01-01
        df = df['1960-01-01':]

        # target series
        target = df[target_col]

        # drop columns with NaN data
        na_cols = df.columns[df.isna().sum() > 0]
        df.drop(na_cols, axis=1, inplace=True)


        # CPI data for last T+1 is not available, but features are -> use for OOS prediction
        oos_sample = df.iloc[-13:-1]
        oos_sample = np.array(oos_sample).reshape((1, oos_sample.shape[0], oos_sample.shape[1]))
        # remove last row (time T) as CPI data for T+1 is not available
        df = df.iloc[:-1]

    return df, target, oos_sample

def _transform_series(x, transformation):
    '''
    This function transforms the series to make them stationary following McCracken and Ng (2015)
    Exception: CPI (CPIAUCSL) is transformed to YoY-change as this is most commonly used

    :param x: a series to be transformed
    :param transformation: the transformation for the given series x
    :return: a transformed (stationary) series
    '''
    if transformation == 1:  # no transformation
        return x
    elif transformation == 2:  # first difference
        return x.diff()
    elif transformation == 3:  # second difference
        return (x.diff()).diff()
    elif transformation == 4:  # log transform
        return np.log(x)
    elif transformation == 5:  # log first diff
        return np.log(x).diff()
    elif transformation == 6:  # log second difference
        return (np.log(x).diff()).diff()
    elif transformation == 7:  # percentage change
        return (x.pct_change()).diff()
    elif transformation == 8:  # YoY-change
        return x.pct_change(12)

def samples(features, target, window_size=12):
    '''
    This function generates samples containing observations t-1 until t-window size for features and observation t for
    target

    :param features: the features (including lagged target variables)
    :param target: the target variable
    :param window_size: how many lagged values to include in the sample
    :return: numpy arrays for features X and target y
    '''
    X, y = [], []

    for i in range(len(features)-window_size+1):
        feat = features[i:i+window_size, :]
        label = target[i+window_size]
        X.append(feat)
        y.append(label)

    return np.array(X), np.array(y)

def plot_fit(fit):
    fig = plt.figure(figsize=(20, 8))
    plt.plot(fit.history['loss'], label='training', color='Blue')
    plt.plot(fit.history['val_loss'], label='validation', color='Red')
    plt.legend()

    return fig


