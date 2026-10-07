import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def train_arch(series, q=1, p=1):
  return arch_model(series, vol="Garch", p=p, q=q, rescale=True).fit(update_freq=0, disp="off")

def apply_features(df, horizon=1):
  df['Returns'] = 100 * df["close_amount"].astype(float).pct_change()
  df.dropna(inplace=True)
  garch_model = train_arch(df["Returns"])
  df["Volatility"] = garch_model.conditional_volatility

  df['Lagged_Return'] = df['Returns'].shift(horizon)
  df['Lagged_Volatility'] = df['Volatility'].shift(horizon)

  df['Target_Vol'] = df['Returns'].abs()
  return df

def _train_garch_ann(X, y, hidden_layer_sizes=(64, 32), alpha=0.01, activation='relu'):

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)

  ann_model = MLPRegressor(
      hidden_layer_sizes=hidden_layer_sizes,
      activation=activation,
      solver='adam',
      alpha=alpha,
      max_iter=10000,
      random_state=42
  )

  ann_model.fit(X_scaled, y)

  return ann_model

def train_garch_ann(df, horizon=1, hidden_layer_sizes=(64, 32), alpha=0.01, activation='relu'):
  df = apply_features(df, horizon=horizon)
  X = df[['Lagged_Return', 'Lagged_Volatility']].values
  y = df['Target_Vol'].values

  ann_model = _train_garch_ann(X, y, hidden_layer_sizes, alpha, activation)

  return ann_model

def predict_garch_ann(df, ann_model, horizon=1):
  df = apply_features(df, horizon=horizon)
  X = df[['Lagged_Return', 'Lagged_Volatility']].values

  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  X_scaled = X_scaled[-horizon:]

  predictions = ann_model.predict(X_scaled)

  return predictions