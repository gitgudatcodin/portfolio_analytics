# Portfolio Analytics (Python edition)

Institutional-grade risk & return characteristics for any stock portfolio —
**no API keys needed**. Prices come from Yahoo Finance via `yfinance`
(split- and dividend-adjusted, i.e. total return).

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## Use it

1. Enter holdings in the sidebar — **Shares** mode (ticker + share count) or
   **Weights %** mode. Edit the table directly, or use **Bulk import**.
2. Pick a history window (1/2/3/5/10Y), a benchmark (SPY/QQQ/DIA/IWM/VTI/AGG/none),
   the risk-free rate, and **buy & hold** vs **rebalanced** sizing.
3. Click **▶ Analyze portfolio**.

## What it computes

CAGR, annualized volatility, Sharpe, Sortino, max drawdown (with peak/trough dates),
Calmar, VaR/CVaR 95%, skew, kurtosis, win rates, best/worst day/month, beta, alpha,
correlation, R², tracking error, information ratio, Treynor, up/down capture,
Euler risk contributions, per-holding stats, correlation matrix — with growth,
drawdown, rolling volatility/beta, histogram, monthly heatmap, and allocation charts.

## Notes

- Data is cached for 1 hour; changing inputs re-runs the analysis.
- Tickers use Yahoo format (`BRK-B`, `VOD.L`); dots are converted automatically.
- Educational analysis, not investment advice.
