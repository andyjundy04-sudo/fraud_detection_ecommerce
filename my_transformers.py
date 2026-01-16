# transformers.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

#OOP transaction_time (object to datetime)
class DateTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, col):
    self.col = col

  def fit(self, X, y=None):
    return self

  def transform (self, X):
    X = X.copy()
    tm = pd.to_datetime(X[self.col], errors='coerce')

    X['month'] = tm.dt.month
    X['day'] = tm.dt.dayofweek
    X['hour'] = tm.dt.hour

    return X.drop(columns=[self.col])

#OOP convert category data (object to frequency)
class FrequencyEncoder(BaseEstimator, TransformerMixin):
  def __init__(self, cols):
    self.cols = cols


  def fit(self, X, y=None):
    self.freq_maps_ = {}
    X = X.copy()

    for col in self.cols:
      self.freq_maps_[col] = X[col].value_counts(normalize=True)

    return self

  def transform(self, X):
    X = X.copy()

    for col in self.cols:
      X[col] = X[col].map(self.freq_maps_[col]).fillna(0)

    return X