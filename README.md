## Quantitative Pair Trading Backtest

A data-driven mean-reversion trading strategy backtested using Python.

## Performance Summary
* **Annualized Return:** 19.23%
* **Sharpe Ratio:** 1.334
* **Backtest Horizon:** 6 Months

## Methodology
1. **Data Sourcing & Cleaning:** Gathered and cleaned historical financial data from the Bloomberg Terminal and Yahoo Finance.
2. **Statistical Testing:** Verified asset co-integration by performing an Augmented Dickey-Fuller (ADF) test between **MU (Micron Technology)** and **AMD (Advanced Micro Devices)** at a 10% significance level.
3. **Strategy:** Developed a mean-reversion model leveraging a rolling Z-score calculated from the pairs' moving average and standard deviation.

## Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Statsmodels (for ADF test)

