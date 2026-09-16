import numpy as np

# ADF Test 
import pandas as pd
import yfinance as yf

# Define list of tickers
y = 'AMD'
tickers_list = ['MU', y]

# Store list in data frame
data = pd.DataFrame(columns = tickers_list)

# Fetch data from yahoo finance
for ticker in tickers_list:
    data[ticker] = yf.download(ticker, '2026-01-05', '2026-07-31', auto_adjust = False)['Adj Close']
    
import statsmodels.api as stat
import statsmodels.tsa.stattools as ts

data = data.dropna()

# Perform ADF test on closing prices of fetched data
X = stat.add_constant(data[y]) # add constant for regression
result = stat.OLS(data['MU'], X).fit()
c_t = ts.adfuller(result.resid)
print(result.resid)
print(c_t)
# Print first 5 rows of fetched data
# print(data['MU'].head())
# print(data[y].head())

# Check co-intergration
if c_t[0] <= c_t[4]['10%'] and c_t[1] <= 0.1:
    print("Pair of securities is co-integrated")
else:
    print("Pair of securities is not co-integrated")
    
    
# calculating rolling mean and rolling std, z score of the spread
alpha = result.params['const']
beta = result.params[y]
spread = result.resid
spread2 = data['MU'] - (alpha +beta*data[y])

# Rolling statistics
window = 20
rolling_mean = spread.rolling(window=window).mean().shift(1) # avoid lookahead bias
rolling_std = spread.rolling(window=window).std().shift(1)

# Z-score of the spread
data['spread'] = spread
data['z_score'] = (spread - rolling_mean) / rolling_std

# print(data[['spread', 'z_score']].tail())

# Position:
#  1 = short spread: sell MU, buy AMD
# -1 = long spread: buy MU, sell AMD
#  0 = no position

position = 0
positions = []
actions = []

for z in data['z_score']:
    action = 'HOLD'
    
    if pd.isna(z):
        positions.append(position)
        actions.append(action)
        continue
    
    if position == 0:
        if z >= 2:
            position = 1
            action = 'ENTER SHORT SPREAD'
            
        elif z <= -2:
            position = -1
            action = 'ENTER LONG SPREAD'
    
    elif position == 1:
        if z >= 3:
            position = 0
            action = 'STOP LOSS SHORT SPREAD'
        elif z <= 0:
            position = 0
            action = 'EXIT SHORT SPREAD'
            
    elif position == -1:
        if z <= -3:
            position = 0
            action = 'STOP LOSS LONG SPREAD'
        elif z >= 0:
            position = 0
            action = 'EXIT LONG SPREAD'
            
    positions.append(position)
    actions.append(action)

data['position'] = positions
data['action'] = actions

print(data[['spread', 'z_score', 'position', 'action']].tail(20))

# Execute on the next trading day
data['executed_position'] = data['position'].shift(1).fillna(0)

# Daily asset returns
data['MU_return'] = data['MU'].pct_change()
data['y_return'] = data[y].pct_change()

# Dollar-neutral normalized weights
mu_weight = 1 / (1 + abs(beta))
y_weight = abs(beta) / (1 + abs(beta))

# Return of a long spread:
# buy MU and sell beta-adjusted AMD
spread_return = (
    mu_weight * data['MU_return']
    - y_weight * data['y_return']
)

data['strategy_return'] = (
    data['executed_position'] * spread_return
)

# Optional transaction costs
transaction_cost = 0.0005  # 0.05% per position unit
turnover = data['executed_position'].diff().abs().fillna(0)
data['strategy_return'] -= turnover * transaction_cost

# Equity curve
data['equity_curve'] = (
     1 + data['strategy_return'].fillna(0)
).cumprod()

# Annualized Sharpe ratio
daily_returns = data['strategy_return'].dropna()

sharpe_ratio = (
    np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    if daily_returns.std() != 0
    else np.nan
)

print(data[['z_score', 'position', 'strategy_return',
            'equity_curve']].tail())

print(f"Total return: {data['equity_curve'].iloc[-1] - 1:.2%}")
print(f"Annualized Sharpe ratio: {sharpe_ratio:.3f}")