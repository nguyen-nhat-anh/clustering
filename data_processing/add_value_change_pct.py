import pandas as pd
import numpy as np


def add_value_change(filename, formula='percentage'):
    """ Add value_change(%) field to dataframe

    :param: filename:str \n
        file name to read data from \n
        the columns are 'DATE', 'CLOSE', 'TICKER', 'OPEN', 'HIGH', 'LOW', 'VOLUME'
    :param: formula:srt, optional, default='percentage' \n
        formula used to compute value_change \n
        'percentage': value_change(t) = (close_price(t) - close_price(t-1))/(close_price(t-1)) \n
        'logarithm': value_change(t) = ln(close_price(t)) - ln(close_price(t-1))
    :return: pandas.core.frame.DataFrame \n
        dataframe with 'VAL_CHANGE(%)' field added
    """
    df = pd.read_csv(filename)
    if formula == 'percentage':
        value_change_pct = (df['CLOSE'][:-1].values / df['CLOSE'][1:].values - 1) * 100
    elif formula == 'logarithm':
        value_change_pct = np.log(df['CLOSE'][:-1].values) - np.log(df['CLOSE'][1:].values)

    result = df[:-1]
    pd.options.mode.chained_assignment = None  # default='warn'
    result['VAL_CHANGE(%)'] = value_change_pct
    return result
