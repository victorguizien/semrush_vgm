import pandas as pd
import warnings
import seaborn as sns
sns.set_style("whitegrid")

warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, message="DataFrameGroupBy.apply")


def calculate_proportions(df, date_col, group_cols, freq='MS'):
    """
    Calculate proportions for multiple groups within a time frequency.

    :param df: pandas DataFrame
    :param date_col: name of the datetime column
    :param group_cols: list of column names to group by (can be more than one column)
    :param freq: frequency for grouping dates (default 'MS' for month start)
    :return: DataFrame with proportions calculated
    """
    df[date_col] = pd.to_datetime(df[date_col])  # Ensure the date column is in datetime format

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(date_col)

    if isinstance(group_cols, str):
        group_cols = [group_cols]

    grouping = [pd.Grouper(freq=freq)] + group_cols

    df_result = df.groupby(grouping).size().reset_index(name='n')

    df_result['n_tot'] = df_result.groupby(pd.Grouper(key=date_col, freq=freq))['n'].transform('sum')

    df_result['prop'] = df_result['n'] / df_result['n_tot']

    return df_result


def calculate_proportions_no_date(df, group_cols, count_col=None):
    """
    Calculate proportions for specified grouping columns.

    :param df: pandas DataFrame
    :param group_cols: list of column names to group by (e.g., ['billingCountry', 'product'])
    :param count_col: name of the column to count (if None, will count rows)
    :return: DataFrame with counts and proportions calculated
    """
    if count_col is None:
        df_result = df.groupby(group_cols).size().reset_index(name='n')
    else:
        df_result = df.groupby(group_cols)[count_col].count().reset_index(name='n')

    total_group_cols = group_cols[:-1]
    df_totals = df_result.groupby(total_group_cols)['n'].sum().reset_index(name='n_tot')

    df_result = df_result.merge(df_totals, on=total_group_cols, how='left')

    df_result['prop'] = df_result['n'] / df_result['n_tot'] * 100

    return df_result


def generate_month_list(start_month, period):
    return [start_month + pd.DateOffset(months=i) for i in range(period)]
