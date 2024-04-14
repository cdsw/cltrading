import pandas as pd
import pandas_ta as ta
from ENV import *
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go

def getStationary(df, fn, scaling=True):
  df['delta'] = (df['close'] / df ['open']) - 1
  df['amplitude'] = (df['high'] / df['low']) - 1
  df['close_over_low'] = (df['close'] / df ['low']) - 1
  del df['open']
  del df['high']
  del df['low']
  df['volume'].replace(0, 0.00001, inplace=True)

  # Scaling
  if scaling:
    df = scale(df, ['delta','amplitude','close_over_low','volume'])

  # Saving
  df.to_csv(f'../data/{fn}-full.csv')
  df.iloc[:500].to_csv(f'../data/{fn}-short.csv')
  df.iloc[:10000].to_csv(f'../data/{fn}-long.csv')
  return df

def scale(df, column_list): #-1 to 1
  for x in column_list:
    scaled = np.log(np.sqrt(df[x] - df[x].min() + 1))
    max_value = scaled.max()
    min_value = scaled.min()
    scaled = 2 * ((scaled - min_value) / (max_value - min_value)) - 1 # -1 to 1
    df[x] = scaled
  return df

def add_ema(df, normalize=False, ma_widths=EMA):
  res = df
  for x in ma_widths:
    alpha = 2 / (x + 1)  # EMA smoothing factor
    ema_values = [df['close'].iloc[0]]  # Start with the first value as the initial EMA

    for i in range(1, len(df['close'])):
      ema = alpha * df['close'].iloc[i] + (1 - alpha) * ema_values[-1]
      ema_values.append(ema)

    res[f'EMA_{x}'] = ema_values
    if normalize:
      res[f'EMA_{x}'] = res[f'EMA_{x}'] / res['close']
  return res

def loadDataSource(dataset_train, TEST):
  if TEST == 'short':
    data_source = f'../data/{dataset_train}-short.csv'
  elif TEST == 'long':
    data_source = f'../data/{dataset_train}-long.csv'
  else:
    data_source = f'../data/{dataset_train}-full.csv'
  return data_source

def getTrainingDivisor(TEST):
  if TEST == 'short':
    TRAINING_DIVISOR = 2
  elif TEST == 'long':
    TRAINING_DIVISOR = 50
  else:
    TRAINING_DIVISOR = 100
  return TRAINING_DIVISOR


def _helperPlotActions(res, color, symbol, size, name):
  x_data = [entry['idx'] for entry in res]
  y_data = [float(entry['price']) for entry in res]
  marker_text = [entry['caption'] for entry in res]
  plotter = go.Scatter(x=x_data, y=y_data, mode='markers+text', marker=dict(color=color, symbol=symbol, size=size), text=marker_text, textposition="top center", name=name)
  return plotter

def translateActionPlot(aset, init_price):
  res_short = []
  res_long = []
  res_close = []
  idx = 0
  for x in aset:
    if x == 'hold':
      pass
    else:
      tx = x.split(' ')
      caption, price = tx[0], tx[3]
      price_clean = float(price)/float(init_price)
      if 'close' in x:
        res_close.append({'caption': "", 'price': price_clean, 'idx': idx})
      elif 'short' in x:
        leverage = caption.split('×')[0]
        res_short.append({'caption': leverage, 'price': price_clean, 'idx': idx})
      elif 'long' in x:
        leverage = caption.split('×')[0]
        res_long.append({'caption': leverage, 'price': price_clean, 'idx': idx})
    idx += 1

  em_dict = dict()
  em_dict['long'] = _helperPlotActions(res_long, '#0e874a', 'triangle-up', 8, "Long")
  em_dict['short'] = _helperPlotActions(res_short, '#ed077a', 'triangle-down', 8, "Short")
  em_dict['close'] = _helperPlotActions(res_close, '#0776ed', 'circle', 6, "Close")

  return em_dict
