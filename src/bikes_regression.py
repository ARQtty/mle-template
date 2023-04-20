from pathlib import Path
import pandas as pd
import pickle
from loguru import logger
from yamlparams import Hparam
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(df.shape)
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


if __name__ == '__main__':
    root = Path('..')
    cfg = Hparam(root / 'config.yaml')

    data = load_data(root / cfg.data.path)
    train, test = preprocess(data)

    model = LinregModel()
    model.fit(train, cfg.model)

    preds = model.predict(test)
    target = test[model.target]

    metrics = model.metrics(preds, target)
    logger.info(metrics)
    model.save(root / cfg.model.path)
