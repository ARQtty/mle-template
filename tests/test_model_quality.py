import pytest
from pathlib import Path
from yamlparams import Hparam
from src.bikes_regression import LinregModel


def test_valid_save():
    root = Path('.')
    cfg = Hparam(root / 'config.yaml')
    model = LinregModel()
    model.load(root / cfg.model.path)
