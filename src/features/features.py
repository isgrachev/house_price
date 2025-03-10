def transform_age(df, remove_yrsold=False):
    """
    Transforms columns with year to columns with age at sale date

    Args:
    ----
    df: pandas dataframe,
        Data
    remove_yrsold: bool, default = False
        Drop column YrSold
    
    Returns:
    --------
    df
        Altered dataframe
    """

    age_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']
    new_names = ['HouseAge', 'RemodAge', 'GarageAge']
    for col, name in zip(age_cols, new_names):
        df[name] = df['YrSold'] - df[col]
        df.drop(columns=[col], inplace=True)
    if remove_yrsold:
        df.drop(columns=['YrSold'], inplace=True)
    return df
