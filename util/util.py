""" Utilities to process dataframe """

import pandas as pd
import numpy as np

def drop_by_iqr_column(df, column, iqr_range=1.5):
    """ drops the observations by finding outliers of given column and given IQR range

    Parameters
    ----------

    df : data frame
        pandas dataframe to be processed.

    column: string
        Column name on which we do outlier processing.

    iqr_range: float
        IQR range out side which we remove the observations. Default 1.5

    Returns
    -------

    df : data frame
        Returns the outlier processed data frame.

    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    IQR = q3 - q1
    q1 = q1 - IQR*iqr_range
    q3 = q3 + IQR*iqr_range
    iqr = (df[column] > q1) & (df[column] < q3)
    return df.loc[iqr]


def drop_by_iqr(df, ignore=[], iqr_range=1.5):
    """ IQR processing by ignoring Y - dependent variable
    
    Parameters
    ----------

    df : data frame
        pandas dataframe to be processed.

    ignore : list of strings
        Columns to be ignored from IQR processing.

    iqr_range: float
        IQR range out side which we remove the observations. Default 1.5

    Returns
    -------

    df : data frame
        Returns the outlier processed data frame.

    """

    cont_vars = df.drop(ignore, axis=1).columns

    for col in cont_vars:
        before = df.shape[0]
        df = drop_by_iqr_column(df, col)
        after = df.shape[0]
        print("On Col:", col, "Dropped:", (before-after))

    return df


def scale_it(df, ignore=[]):
    """ Apply standard scalar on data frame by ignoring columns

    Parameters
    ----------

    df : data frame
        pandas data frame to operate on.

    ignore: list of strings
        Columns to be ignored while scaling.

    Returns
    -------

    df : data frame
        Returns the scaled data frame.

    """
    from sklearn.preprocessing import StandardScaler
    std_scale = StandardScaler()
    for col in df.columns.values:
        if col not in ignore:
            df[col] = std_scale.fit_transform(df[[col]])

    return df

def split_dataframe(df, Y, scale=False, scale_ignore=[], factor=0.7, seed=7):
    """ Split the given data frame into training and data sets.

    Parameters
    ----------

    df : data frame
        pandas data frame to operate on.

    Y : string
        Dependent variable

    scale : boolean (default=False)
        flag to enable data scaling/normalization.

    scale_ignore: list of strings
        Columns to be ignored while scaling.

    factor : float (default=0.7)
        Split factor of training and data set.

    seed : int
        random seed value while split.

    Returns
    -------

    (X_train, Y_train, X_test, Y_test) : tuple
        Tuple of training and test nunmpy arrays of dependent and independent varibale sets.

    """

    # Add Y to scale_ignore
    scale_ignore += [Y]

    if scale:
        df = scale_it(df, scale_ignore)

    np.random.seed(seed=seed)
    msk = np.random.rand(len(df)) < factor
    train = df[msk]
    test = df[~msk]

    X_train = train.drop(Y, axis=1).values
    Y_train = train[Y].values

    X_test = test.drop(Y, axis=1).values
    Y_test = test[Y].values

    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    return (X_train, Y_train, X_test, Y_test)
