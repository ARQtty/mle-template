import pandas as pd
from sqlalchemy import create_engine
import vault_cli


if __name__ == '__main__':
    vault = vault_cli.client.get_client(token=open('vault/token-file.txt').read(), url='http://vault:8200')
    secrets = vault.get_secrets(path='cubbyhole/db')['cubbyhole/db']

    user = secrets['pg_user']
    password = secrets['pg_password']
    db = secrets['pg_db']
    engine = create_engine(f"postgresql://{user}:{password}@database:5432/{db}")

    df = pd.read_csv('data/raw/bike-sharing-demand/train.csv')
    df.to_sql('bikes', engine)
