import os
import pandas as pd
from sqlalchemy import create_engine


if __name__ == '__main__':
    user = os.getenv('POSTGRES_USER')
    password = os.getenv('POSTGRES_PASSWORD')
    db = os.getenv('POSTGRES_DB')
    engine = create_engine(f'postgresql://{user}:{password}@database:5432/{db}')

    df = pd.read_csv('data/raw/bike-sharing-demand/train.csv')
    df.to_sql('bikes', engine)
