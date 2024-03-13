INITIAL_ACCOUNT_BALANCE = 100000
WINDOW_SIZE = 1
EMA = [8,28,88]

FEATURES = ['close', 'volume', 'delta', 'amplitude', 'close_over_low', ]
for e in EMA:
  FEATURES.append(f'EMA_{e}')

LEN_FEATURES = len(FEATURES)
ACTIONS = ['long', 'short', 'hold', 'close']
LEVERAGE = [1]
TRADING_FEE_OPEN = 0.0005 # portion
TRADING_FEE_CLOSE = 0.0005 # portion
MINIMUM_TRANSACTION_DOLLAR = 5
PRECISION = 8 # precision for unit

INTERVAL = 100000
EPISODE = 5