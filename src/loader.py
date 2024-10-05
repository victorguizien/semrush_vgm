import pandas as pd
import os
from dotenv import load_dotenv
from src.processing import generate_month_list


current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, 'config', '.env')
load_dotenv(dotenv_path=env_path)

data_path = os.getenv('DATA_PATH')


def load_and_clean():
    """
    Load and clean transaction data from a CSV file.

    The function performs the following operations:
    - Loads data from 'synthetical_payments.csv'
    - Converts transaction times to datetime
    - Removes data from 2018 onwards
    - Eliminates transactions with amounts >= 10000
    - Removes duplicate transactions
    - Expands annual subscriptions into monthly data

    Returns:
    tuple: (cleaned DataFrame with monthly data, original transaction DataFrame)
           Returns (None, None) if the file doesn't exist.
    """

    print("Loading and cleaning data..")

    file_path = f'{data_path}/synthetical_payments.csv'

    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep=',')
        df['datetime'] = pd.to_datetime(df['transactionTime'], unit='s')

        # Eliminate single point of 2018
        df = df.loc[df['datetime'] < '2018-01-01']

        # Eliminate absurd values
        df = df.loc[~(df['amount'] >= 10000)]

        # Elimiante duplicate transactions
        df_check = df.set_index('datetime').groupby([pd.Grouper(freq='MS'), 'userId']).size().reset_index().rename(columns={0: 'n'})
        ids_duplicated = df_check.loc[df_check['n'] > 1]['userId']
        indices_to_delete = df.loc[df['userId'].isin(ids_duplicated)].groupby('userId').head(1).index

        mask = ~df.index.isin(indices_to_delete)
        df = df.iloc[mask]

        df_transactions = df.copy()
        # Explode anual users into months
        df["mrr_month"] = df.apply(lambda row: generate_month_list(row["datetime"], row["period"]), axis=1)
        df["mrr"] = df["amount"] / df["period"]
        df = df.explode("mrr_month").reset_index(drop=True)

        # Eliminate again 2018 as we only have monthly data from exploded users
        df = df.loc[df['mrr_month'] < '2018-01-01']

        print("Done.")
    else:
        print("Error. File path doesn't exist")
        df = None
        df_transactions = None

    return df, df_transactions
