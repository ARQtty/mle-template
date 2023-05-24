import os
from pathlib import Path
import pandas as pd
import pickle
from loguru import logger
from sqlalchemy import create_engine
from yamlparams import Hparam
from confluent_kafka import Producer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import vault_cli


def load_data(table: str) -> pd.DataFrame:
    vault = vault_cli.client.get_client(token=open('vault/token-file.txt').read(), url='http://vault:8200')
    secrets = vault.get_secrets(path='cubbyhole/db')['cubbyhole/db']

    user = secrets['pg_user']
    password = secrets['pg_password']
    db = secrets['pg_db']
    host = 'database'
    engine = create_engine(f'postgresql://{user}:{password}@{host}:5432/{db}')

    df = pd.read_sql_query(f'select * from {table}', con=engine)
    logger.debug(df.shape)
    return df


def preprocess(df: pd.DataFrame) -> list[pd.DataFrame]:
    df = df.drop('weather', axis=1).join(pd.get_dummies(df.weather, prefix='weather'))
    df = df.drop('season', axis=1).join(pd.get_dummies(df.season, prefix='season'))
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    return [train, test]


class LinregModel:
    def __init__(self):
        self.features = [
            'holiday',
            'workingday',
            'weather_1',
            'weather_2',
            'weather_3',
            'weather_4',
            'season_1',
            'season_2',
            'season_3',
            'season_4',
            'temp',
            'atemp',
            'humidity',
            'windspeed'
        ]

        self.target = 'count'

    def fit(self, train: pd.DataFrame, config: Hparam):
        self.model = LinearRegression()
        self.model.fit(train[self.features], train[self.target])

    def predict(self, df: pd.DataFrame) -> pd.Series:
        preds = self.model.predict(df[self.features])
        return preds

    def metrics(self, preds: pd.Series, gt: pd.Series) -> dict[str, float]:
        metrics = dict(
            mae=mean_absolute_error(gt, preds),
            mape=mean_absolute_percentage_error(gt, preds)
        )
        return metrics

    def save(self, path: Path):
        pickle.dump(self.model, open(path, 'wb'))

    def load(self, path: Path):
        self.model = pickle.load(open(path, 'rb'))


def kafka_delivery_report(err, msg):
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))


if __name__ == '__main__':
    logger.info('started')
    root = Path('.')
    cfg = Hparam(root / 'config.yaml')

    logger.info('loading data')
    data = load_data(cfg.data.table)
    train, test = preprocess(data)

    model = LinregModel()
    model.fit(train, cfg.model)

    preds = model.predict(test)
    target = test[model.target]

    metrics = model.metrics(preds, target)
    logger.info(metrics)
    model.save(root / cfg.model.path)

    # predictions to kafka
    host = 'kafka'
    # create topic
    os.system(f"kafka_2.13-3.4.0/bin/kafka-topics.sh --create --topic topic --bootstrap-server {host}:29092 --if-not-exists")

    # write data
    logger.info('sending to kafka')
    p = Producer({'bootstrap.servers': f'{host}:29092'})
    for pred, tgt in zip(preds, target):
        p.poll(0)
        msg = str(dict(predicted_price=pred, acutal_price=tgt, diff=round(tgt - pred, 1)))
        p.produce('topic', msg.encode('utf-8'))
    p.flush()

    # consume messages
    os.system(f"kafka_2.13-3.4.0/bin/kafka-console-consumer.sh --topic topic --from-beginning --bootstrap-server {host}:29092")
