import pandas as pd

def add_value_change(filename):
    """ Add value_change(%) field to dataframe 
    
    Parameters
    ----------
    filename : str
        file name to read data from
        the columns are 'DATE', 'CLOSE', 'TICKER', 'OPEN', 'HIGH', 'LOW', 'VOLUME'

    Returns
    -------
    : pandas.core.frame.DataFrame
        dataframe with value_change(%) field added
    """
    df = pd.read_csv(filename)
    value_change_pct = (df['CLOSE'][:-1].values / df['CLOSE'][1:].values - 1) * 100
    df_1month = df[:-1]
    df_1month['VAL_CHANGE(%)'] = value_change_pct
    return df_1month
