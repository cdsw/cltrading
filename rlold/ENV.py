INITIAL_ACCOUNT_BALANCE = 100000
WINDOW_SIZE = 5
FEATURES = ['close', 'volume', 'delta', 'amplitude', 'close_over_low']
LEN_FEATURES = len(FEATURES)
ACTIONS = ['long', 'short', 'hold', 'close']
LEVERAGE = [1]
TRADING_FEE_OPEN = 0.0005 # portion
TRADING_FEE_CLOSE = 0.0005 # portion
MINIMUM_TRANSACTION_DOLLAR = 5
PRECISION = 8 # precision for unit

REWARD = {'open': 5, 'hold': 1, 'close': 1, 'insufficient': -1}

INTERVAL = 100000