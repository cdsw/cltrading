import pandas as pd
import pandas_ta as ta
from ENV import *

def add_ema(df, ma_widths=EMA):
    res = [df]
    for x in ma_widths:
      alpha = 2 / (x + 1)  # EMA smoothing factor
      ema_values = [df['close'].iloc[0]]  # Start with the first value as the initial EMA

      for i in range(1, len(df['close'])):
        ema = alpha * df['close'].iloc[i] + (1 - alpha) * ema_values[-1]
        ema_values.append(ema)

      df[f'EMA_{x}'] = ema_values
    return df


# df = pd.DataFrame({
#   'no': [1,2,3,4,5,6,7,8],
#   'xdata': [1,10,20,30,40,50,60,70]
# })

# df = add_ema(df,[5])

# print(df)
