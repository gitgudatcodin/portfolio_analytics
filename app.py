"""
Portfolio Analytics — institutional-grade risk & return characteristics for any
stock portfolio. No API keys: prices come from Yahoo Finance via yfinance
(split- and dividend-adjusted, i.e. total return).

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""
import re
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Portfolio Analytics", page_icon="◈", layout="wide")

# ---------------------------------------------------------------- dark theme
st.markdown("""
<style>
  :root{--accent:#38bdf8;}
  .stApp{background:#0b0e14;}
  section[data-testid="stSidebar"]{background:#0e1219;}
  h1,h2,h3{color:#e8ecf4 !important;}
  .kpi{background:#121826;border:1px solid #232b3d;border-radius:12px;padding:14px 16px;}
  .kpi .l{font-size:11px;text-transform:uppercase;letter-spacing:.8px;color:#8b93a9;}
  .kpi .v{font-size:26px;font-weight:700;color:#e8ecf4;font-variant-numeric:tabular-nums;}
  .kpi .s{font-size:12px;color:#8b93a9;}
  .pos{color:#34d399;} .neg{color:#f87171;}
  .src-note{color:#8b93a9;font-size:12.5px;}
  div[data-testid="stMetric"]{background:#121826;border:1px solid #232b3d;border-radius:12px;padding:12px 14px;}
</style>
""", unsafe_allow_html=True)

ANN = 252
PALETTE = ["#38bdf8", "#2dd4bf", "#fbbf24", "#f472b6", "#a78bfa",
           "#34d399", "#fb923c", "#94a3b8", "#f87171", "#22d3ee"]

# ---------------------------------------------------------------- formatting
def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{x*100:.{d}f}%"

def num2(x):
    return "—" if x is None or not np.isfinite(x) else f"{x:.2f}"

def cls(x):
    if x is None or not np.isfinite(x):
        return ""
    return "pos" if x > 0 else ("neg" if x < 0 else "")

def kpi_card(label, value, sub="", vcls=""):
    st.markdown(
        f'<div class="kpi"><div class="l">{label}</div>'
        f'<div class="v {vcls}">{value}</div><div class="s">{sub}</div></div>',
        unsafe_allow_html=True)

def norm_ticker(t):
    return t.strip().upper().replace(".", "-").replace(":", "-")

# ---------------------------------------------------------------- data layer
def _safe_download(tickers, start_iso, end_iso):
    try:
        return yf.download(list(tickers), start=start_iso, end=end_iso,
                           auto_adjust=True, progress=False, threads=True)
    except Exception:
        return None


def _close_frame(df, tickers):
    """Extract a ticker-keyed Close DataFrame, robust to yfinance column layouts."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = set(df.columns.get_level_values(0))
        if "Close" in lvl0:                      # (Field, Ticker) layout
            closes = df["Close"]
        else:                                    # (Ticker, Field) layout
            closes = df.xs("Close", axis=1, level=1)
    else:                                        # single ticker, flat columns
        closes = df[["Close"]] if "Close" in df.columns else pd.DataFrame()
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(tickers[0] if tickers else "Close")
    return closes


@st.cache_data(show_spinner=False, ttl=3600)
def download_prices(tickers, start_iso, end_iso):
    """Daily closes via yfinance (auto-adjusted). Returns (closes_df, failed).

    One batched request for all tickers; any ticker that comes back empty is
    retried individually (Yahoo intermittently drops single tickers)."""
    tickers = list(tickers)
    closes = _close_frame(_safe_download(tickers, start_iso, end_iso), tickers)
    closes = closes.dropna(axis=1, how="all")
    for t in [t for t in tickers if t not in closes.columns]:
        s = None
        for _ in range(3):
            one = _close_frame(_safe_download([t], start_iso, end_iso), [t])
            if not one.empty and t in one.columns and one[t].notna().any():
                s = one[t]
                break
            time.sleep(2)
        if s is not None:
            closes[t] = s
    closes = closes[[t for t in tickers if t in closes.columns]]
    failed = [t for t in tickers if t not in closes.columns]
    closes = closes.dropna(how="any")
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    return closes, failed

BENCHMARKS = {
    "SPY — S&P 500": "SPY", "QQQ — Nasdaq 100": "QQQ", "DIA — Dow 30": "DIA",
    "IWM — Russell 2000": "IWM", "VTI — Total US market": "VTI",
    "AGG — US bonds": "AGG", "None": "",
}

# ---------------------------------------------------------------- engine
def _clean(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]

def skewness(x):
    x = _clean(x)
    if len(x) < 3:
        return np.nan
    s = x.std(ddof=0)
    return np.mean(((x - x.mean()) / s) ** 3) if s > 0 else np.nan

def kurtosis(x):  # excess
    x = _clean(x)
    if len(x) < 4:
        return np.nan
    s = x.std(ddof=0)
    return np.mean(((x - x.mean()) / s) ** 4) - 3 if s > 0 else np.nan

def max_drawdown(equity):
    eq = np.asarray(equity, dtype=float)
    peak = eq[0]; peak_i = 0; md = 0.0; trough_i = 0; peak_at_trough = 0
    for i in range(1, len(eq)):
        if eq[i] > peak:
            peak = eq[i]; peak_i = i
        dd = (peak - eq[i]) / peak
        if dd > md:
            md = dd; trough_i = i; peak_at_trough = peak_i
    return md, peak_at_trough, trough_i

def portfolio_returns(rets, weights, mode):
    rets = np.asarray(rets, dtype=float)
    w = np.asarray(weights, dtype=float)
    if mode == "fixed":
        return rets @ w
    v = w.copy()
    out = np.empty(len(rets))
    for t in range(len(rets)):
        tot = v.sum()
        out[t] = (v / tot) @ rets[t]
        v = v * (1 + rets[t])
    return out

def portfolio_metrics(pr, br, rf_annual):
    pr = _clean(pr)
    n = len(pr)
    rf_d = (1 + rf_annual) ** (1 / ANN) - 1
    m = pr.mean()
    vol = pr.std(ddof=1) * np.sqrt(ANN)
    eq = 10000 * np.cumprod(1 + pr)
    eq = np.insert(eq, 0, 10000)
    total = eq[-1] / eq[0] - 1
    cagr = (1 + total) ** (ANN / n) - 1
    sharpe = (m - rf_d) / pr.std(ddof=1) * np.sqrt(ANN) if vol > 1e-12 else np.nan
    dn = pr[pr < 0]
    down_dev = np.sqrt(np.mean(dn ** 2)) * np.sqrt(ANN) if len(dn) > 1 else 0.0
    sortino = (cagr - rf_annual) / down_dev if down_dev > 0 else np.nan
    dd, pk, tr = max_drawdown(eq)
    calmar = cagr / dd if dd > 0 else np.nan
    var95 = -np.quantile(pr, 0.05, method="linear")
    tail = pr[pr <= -var95]
    cvar95 = -tail.mean() if len(tail) else np.nan
    out = dict(n=n, total_ret=total, cagr=cagr, vol=vol, sharpe=sharpe,
               sortino=sortino, down_dev=down_dev, max_dd=dd, dd_peak=pk,
               dd_trough=tr, calmar=calmar, var95=var95, cvar95=cvar95,
               win_rate=np.mean(pr > 0), skew=skewness(pr), kurt=kurtosis(pr),
               best_day=pr.max(), worst_day=pr.min(), equity=eq)
    if br is not None and len(br) == n:
        br = _clean(br)
        bv = np.cov(br, ddof=1)
        beta = np.cov(pr, br, ddof=1)[0, 1] / bv if bv > 0 else np.nan
        alpha = (m - beta * br.mean()) * ANN
        corr = np.corrcoef(pr, br)[0, 1]
        ex = pr - br
        te = ex.std(ddof=1) * np.sqrt(ANN)
        info = ex.mean() * ANN / te if te > 0 else np.nan
        up_m = br > 0; dn_m = br < 0
        up_cap = pr[up_m].mean() / br[up_m].mean() if up_m.any() and br[up_m].mean() != 0 else np.nan
        dn_cap = pr[dn_m].mean() / br[dn_m].mean() if dn_m.any() and br[dn_m].mean() != 0 else np.nan
        treynor = (cagr - rf_annual) / beta if np.isfinite(beta) and beta != 0 else np.nan
        beq = 10000 * np.cumprod(1 + br)
        out.update(dict(beta=beta, alpha=alpha, corr=corr, r2=corr ** 2, te=te,
                        info=info, up_cap=up_cap, dn_cap=dn_cap, treynor=treynor,
                        bench_total=beq[-1] / 10000 - 1,
                        bench_cagr=(beq[-1] / 10000) ** (ANN / n) - 1,
                        bench_vol=br.std(ddof=1) * np.sqrt(ANN)))
    return out

def holdings_metrics(rets, weights, port_rets, rf_annual):
    rets = np.asarray(rets, dtype=float)
    w = np.asarray(weights, dtype=float)
    n_a = rets.shape[1]
    cov_m = np.cov(rets, rowvar=False, ddof=1)
    mc = cov_m @ w                       # marginal contribution to daily variance
    pw = rets @ w                        # fixed-weight portfolio returns (target allocation)
    dvol = pw.std(ddof=1)                # Euler identity: sum = w'Σw / σ² = 1
    rf_d = (1 + rf_annual) ** (1 / ANN) - 1
    pv = np.var(port_rets, ddof=1)
    out = []
    for i in range(n_a):
        rs = rets[:, i]
        m = rs.mean(); v = rs.std(ddof=1) * np.sqrt(ANN)
        tr = np.prod(1 + rs) - 1
        cagr = (1 + tr) ** (ANN / len(rs)) - 1
        sharpe = (m - rf_d) / rs.std(ddof=1) * np.sqrt(ANN) if v > 0 else np.nan
        beta = np.cov(rs, port_rets, ddof=1)[0, 1] / pv if pv > 0 else np.nan
        risk = w[i] * mc[i] / (dvol ** 2) if dvol > 1e-12 else np.nan  # Euler share of vol
        out.append(dict(cagr=cagr, vol=v, sharpe=sharpe, beta=beta,
                        ret_contrib=w[i] * cagr, risk_contrib=risk, total_ret=tr))
    return out

def monthly_returns(dates, pr):
    s = pd.Series(pr, index=pd.to_datetime(dates))
    g = (1 + s).resample("ME").prod() - 1
    return list(zip(g.index.strftime("%Y-%m"), g.values))

# ---------------------------------------------------------------- charts
def _layout(fig, title, ytitle=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e8ecf4")),
        paper_bgcolor="#121826", plot_bgcolor="#121826",
        font=dict(color="#8b93a9", size=11),
        xaxis=dict(gridcolor="#1c2333", zerolinecolor="#232b3d"),
        yaxis=dict(title=ytitle, gridcolor="#1c2333", zerolinecolor="#232b3d"),
        margin=dict(l=10, r=10, t=44, b=10), legend=dict(font=dict(size=11)),
        hovermode="x unified")
    return fig

def growth_chart(dates, port_eq, bench_eq=None, bench_name=""):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=port_eq, name="Portfolio",
                             line=dict(color="#38bdf8", width=2)))
    if bench_eq is not None:
        fig.add_trace(go.Scatter(x=dates, y=bench_eq, name=bench_name,
                                 line=dict(color="#8b93a9", width=1.5, dash="dash")))
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return _layout(fig, "Growth of $10,000", "Value ($)")

def donut_chart(labels, weights):
    fig = go.Figure(go.Pie(labels=labels, values=weights, hole=0.55,
                           marker=dict(colors=PALETTE),
                           textinfo="label+percent", textfont=dict(size=11)))
    fig.update_layout(paper_bgcolor="#121826", font=dict(color="#e8ecf4", size=11),
                      title=dict(text="Allocation", font=dict(size=15, color="#e8ecf4")),
                      margin=dict(l=10, r=10, t=44, b=10), showlegend=False)
    return fig

def underwater_chart(dates, equity):
    eq = np.asarray(equity)
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq) * 100
    fig = go.Figure(go.Scatter(x=dates, y=dd, fill="tozeroy", name="Drawdown",
                               line=dict(color="#f87171", width=1.5),
                               fillcolor="rgba(248,113,113,.25)"))
    fig.update_yaxes(ticksuffix="%")
    return _layout(fig, "Underwater chart (drawdown)", "Drawdown %")

def rolling_vol_chart(dates, pr, win=63):
    rv = pd.Series(pr).rolling(win).std(ddof=1) * np.sqrt(ANN) * 100
    fig = go.Figure(go.Scatter(x=dates, y=rv, name=f"{win}d vol",
                               line=dict(color="#fbbf24", width=1.5)))
    fig.update_yaxes(ticksuffix="%")
    return _layout(fig, f"Rolling {win}-day volatility", "Ann. vol %")

def rolling_beta_chart(dates, pr, br, win=126):
    p = pd.Series(pr); b = pd.Series(br)
    beta = p.rolling(win).cov(b) / b.rolling(win).var(ddof=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=beta, name=f"{win}d beta",
                             line=dict(color="#a78bfa", width=1.5)))
    fig.add_hline(y=1, line_dash="dash", line_color="#8b93a9", line_width=1)
    return _layout(fig, f"Rolling {win}-day beta vs benchmark", "Beta")

def hist_chart(pr):
    pr = _clean(pr) * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=pr, nbinsx=60, name="Daily returns",
                               marker=dict(color="#38bdf8", opacity=0.65)))
    m, s = pr.mean(), pr.std(ddof=1)
    xs = np.linspace(pr.min(), pr.max(), 200)
    pdf = (1 / (s * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - m) / s) ** 2)
    scale = len(pr) * (pr.max() - pr.min()) / 60
    fig.add_trace(go.Scatter(x=xs, y=pdf * scale, name="Normal fit",
                             line=dict(color="#f472b6", width=2)))
    fig.update_xaxes(ticksuffix="%")
    return _layout(fig, "Distribution of daily returns", "Frequency")

def heatmap_fig(monthly):
    # monthly: list of (ym, r)
    df = pd.DataFrame(monthly, columns=["ym", "r"])
    df["y"] = df["ym"].str[:4]; df["m"] = df["ym"].str[5:7].astype(int)
    piv = df.pivot(index="y", columns="m", values="r") * 100
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    piv = piv.reindex(columns=range(1, 13))
    z = piv.values
    text = np.where(np.isnan(z), "", np.round(z, 1).astype(str))
    fig = go.Figure(go.Heatmap(z=z, x=months, y=piv.index, text=text,
                               texttemplate="%{text}", textfont=dict(size=10),
                               colorscale=[[0, "#7f1d1d"], [0.5, "#1c2333"], [1, "#065f46"]],
                               zmid=0, hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>"))
    fig.update_layout(paper_bgcolor="#121826", font=dict(color="#8b93a9"),
                      title=dict(text="Monthly returns", font=dict(size=15, color="#e8ecf4")),
                      margin=dict(l=10, r=10, t=44, b=10),
                      xaxis=dict(side="top"))
    return fig

def risk_bar_fig(labels, risk_contrib):
    rc = np.array(risk_contrib) * 100
    fig = go.Figure(go.Bar(x=labels, y=rc, marker_color=PALETTE[:len(labels)],
                           text=[f"{v:.1f}%" for v in rc], textposition="outside"))
    fig.update_yaxes(ticksuffix="%")
    return _layout(fig, "Risk contribution (share of portfolio volatility)", "% of vol")

def corr_fig(labels, rets):
    c = pd.DataFrame(rets, columns=labels).corr()
    fig = px.imshow(c.values, x=labels, y=labels, text_auto=".2f",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(paper_bgcolor="#121826", font=dict(color="#8b93a9", size=11),
                      title=dict(text="Holdings correlation", font=dict(size=15, color="#e8ecf4")),
                      margin=dict(l=10, r=10, t=44, b=10), coloraxis_showscale=False)
    return fig

# ---------------------------------------------------------------- sidebar
st.sidebar.markdown("## ◈ Portfolio Analytics")
st.sidebar.caption("Institutional-grade risk & return characteristics for any stock portfolio. Prices via Yahoo Finance — no API keys.")

if "hold_df" not in st.session_state:
    st.session_state.hold_df = pd.DataFrame(
        {"Ticker": ["AAPL", "MSFT", "NVDA", "JNJ", "XOM", "JPM"],
         "Amount": [20, 15, 30, 25, 40, 20]})
if "ed_key" not in st.session_state:
    st.session_state.ed_key = 0

def reset_holdings(df):
    # Rebuild with explicit dtypes: an empty frame from [] infers float64,
    # which breaks the TextColumn config and kills the editor (Clear bug).
    st.session_state.hold_df = pd.DataFrame({
        "Ticker": pd.Series(list(df["Ticker"]), dtype="str"),
        "Amount": pd.to_numeric(df["Amount"], errors="coerce").astype("float64"),
    })
    st.session_state.ed_key += 1
    st.rerun()

st.sidebar.subheader("Holdings")
mode = st.sidebar.radio("Input mode", ["Shares", "Weights %"], horizontal=True,
                        help="Shares sizes positions by market value; Weights % uses the percentages directly.")
hold_df = st.sidebar.data_editor(
    st.session_state.hold_df, num_rows="dynamic", use_container_width=True,
    column_config={"Ticker": st.column_config.TextColumn("Ticker", max_chars=10),
                   "Amount": st.column_config.NumberColumn("Amount", min_value=0)},
    key=f"hold_editor_{st.session_state.ed_key}")
c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("+ Sample", use_container_width=True):
        reset_holdings(pd.DataFrame(
            {"Ticker": ["AAPL", "MSFT", "NVDA", "JNJ", "XOM", "JPM"],
             "Amount": [20, 15, 30, 25, 40, 20]}))
with c2:
    if st.button("Clear", use_container_width=True):
        reset_holdings(pd.DataFrame({"Ticker": [], "Amount": []}))

with st.sidebar.expander("Bulk import"):
    bulk = st.text_area("One per line: TICKER, amount", "AAPL, 20\nMSFT, 15",
                        height=90, label_visibility="collapsed")
    if st.button("Import lines", use_container_width=True):
        rows = []
        for line in bulk.splitlines():
            p = [x.strip() for x in re.split(r"[,;\t]", line) if x.strip()]
            if len(p) >= 2 and re.match(r"^[A-Za-z.\-:]{1,10}$", p[0]):
                try:
                    rows.append((p[0].upper(), float(p[1])))
                except ValueError:
                    pass
        if rows:
            reset_holdings(pd.DataFrame(rows, columns=["Ticker", "Amount"]))

st.sidebar.subheader("Analysis settings")
years = st.sidebar.selectbox("Historical data window", [1, 2, 3, 5, 10], index=2,
                             format_func=lambda y: f"{y} year{'s' if y > 1 else ''}")
bench_label = st.sidebar.selectbox("Benchmark", list(BENCHMARKS.keys()), index=0)
bench = BENCHMARKS[bench_label]
rf = st.sidebar.number_input("Risk-free rate %", value=4.0, step=0.1, min_value=0.0, max_value=20.0) / 100
sizing = st.sidebar.selectbox("Position sizing over history",
                              ["Buy & hold (weights drift)", "Rebalanced (fixed weights)"])
sizing_mode = "drift" if sizing.startswith("Buy") else "fixed"
st.sidebar.caption("Buy & hold lets winners grow their weight, like a real untouched portfolio.")

analyze = st.sidebar.button("▶ Analyze portfolio", type="primary", use_container_width=True)

# ---------------------------------------------------------------- analysis
def run_analysis(df, years, bench, rf, sizing_mode, mode):
    recs = []
    for _, r in df.iterrows():
        t = str(r["Ticker"]).strip()
        try:
            a = float(r["Amount"])
        except (TypeError, ValueError):
            continue
        if t and np.isfinite(a) and a > 0:
            recs.append((norm_ticker(t), a))
    if not recs:
        return None, "Add at least one holding with a ticker and a positive amount."
    holdings = recs
    need = list(dict.fromkeys([t for t, _ in holdings] + ([bench] if bench else [])))
    end = date.today()
    start = end - timedelta(days=int(years * 366 + 10))
    with st.spinner(f"Downloading {len(need)} price series from Yahoo Finance…"):
        closes, failed = download_prices(tuple(need), start.isoformat(),
                                         (end + timedelta(days=1)).isoformat())
    if closes is None or closes.empty:
        return None, f"Could not download price data for: {', '.join(need)}. Check tickers / connection."
    failed_hold = [t for t in failed if t in [h[0] for h in holdings]]
    ok_hold = [(t, a) for t, a in holdings if t in closes.columns]
    if not ok_hold:
        return None, "Could not download price data for any holding."
    # align to common trading days
    sub = closes[[t for t, _ in ok_hold] + ([bench] if bench and bench in closes.columns else [])]
    dates = sub.index.strftime("%Y-%m-%d").tolist()
    rets = sub.pct_change().iloc[1:].values
    ret_dates = dates[1:]
    latest = sub.iloc[-1].values
    n_h = len(ok_hold)
    if mode == "Weights %":
        tot = sum(a for _, a in ok_hold) or 1
        weights = np.array([a / tot for _, a in ok_hold])
    else:
        vals = np.array([a * latest[i] for i, (_, a) in enumerate(ok_hold)])
        weights = vals / (vals.sum() or 1)
    port_rets = portfolio_returns(rets[:, :n_h], weights, sizing_mode)
    bench_rets = rets[:, n_h] if (bench and bench in closes.columns) else None
    M = portfolio_metrics(port_rets, bench_rets, rf)
    H = holdings_metrics(rets[:, :n_h], weights, port_rets, rf)
    monthly = monthly_returns(ret_dates, port_rets)
    bench_eq = (10000 * np.cumprod(1 + bench_rets)) if bench_rets is not None else None
    # full benchmark metric set, so every displayed number has a benchmark twin
    BM = portfolio_metrics(bench_rets, None, rf) if bench_rets is not None else None
    bench_monthly = monthly_returns(ret_dates, bench_rets) if bench_rets is not None else None
    return dict(holdings=[t for t, _ in ok_hold], weights=weights, dates=dates,
                ret_dates=ret_dates, port_rets=port_rets, bench_rets=bench_rets,
                rets_hold=rets[:, :n_h],
                M=M, BM=BM, H=H, monthly=monthly, bench_monthly=bench_monthly,
                bench_eq=bench_eq, bench=bench,
                bench_label=bench_label, rf=rf, years=years, sizing_mode=sizing_mode,
                failed_hold=failed_hold,
                bench_failed=bool(bench and bench not in closes.columns)), None

if analyze:
    res, err = run_analysis(hold_df, years, bench, rf, sizing_mode, mode)
    if err:
        st.error(err)
    else:
        st.session_state["res"] = res
        if res["failed_hold"]:
            st.warning(f"No data for: {', '.join(res['failed_hold'])} — excluded from the analysis.")
        if res["bench_failed"]:
            st.warning(f"No data for benchmark {bench} — benchmark comparisons skipped.")

# ---------------------------------------------------------------- main
res = st.session_state.get("res")
if not res:
    st.title("Portfolio characteristics")
    st.info("Enter holdings in the sidebar and click **Analyze portfolio**.")
    st.markdown('<p class="src-note">Prices: Yahoo Finance via yfinance — split- and dividend-adjusted '
                '(total return). No API keys needed.</p>', unsafe_allow_html=True)
    st.stop()

M, H = res["M"], res["H"]
tickers, w = res["holdings"], res["weights"]
pr, br, dates, rdates = res["port_rets"], res["bench_rets"], res["dates"], res["ret_dates"]
has_b = br is not None and np.isfinite(M.get("beta", np.nan))
BM = res["BM"]

def bmk(key, fm=pct):
    """'Bench X: value' twin for any metric, or a no-benchmark note."""
    return f"Bench {res['bench']}: {fm(BM[key])}" if has_b else "no benchmark"

st.title("Portfolio characteristics")
st.markdown(f'<p class="src-note">{len(tickers)} holdings · {res["years"]}Y window · '
            f'{rdates[0]} → {rdates[-1]} · {"rebalanced" if res["sizing_mode"]=="fixed" else "buy & hold"} · '
            f'Yahoo Finance (adj. closes)</p>', unsafe_allow_html=True)

tab_ov, tab_pf, tab_rk, tab_hd = st.tabs(["Overview", "Performance", "Risk", "Holdings"])

with tab_ov:
    c = st.columns(4)
    with c[0]: kpi_card("CAGR", pct(M["cagr"]), f"{res['years']}Y ann. · " + bmk("cagr"), cls(M["cagr"]))
    with c[1]: kpi_card("Volatility", pct(M["vol"]), "ann. σ · " + bmk("vol"))
    with c[2]: kpi_card("Sharpe", num2(M["sharpe"]), f"rf {res['rf']*100:.1f}% · " + bmk("sharpe", num2), cls(M["sharpe"]))
    with c[3]: kpi_card("Max drawdown", pct(M["max_dd"]), "peak → trough · " + bmk("max_dd"), "neg")
    c = st.columns(4)
    with c[0]: kpi_card("Beta", num2(M.get("beta")), f"vs {res['bench']}" if has_b else "no benchmark")
    with c[1]: kpi_card("Alpha", pct(M.get("alpha")), f"annualized, vs {res['bench'] if has_b else '—'}", cls(M.get("alpha")))
    with c[2]: kpi_card("VaR 95%", pct(M["var95"]), "daily · " + bmk("var95"), "neg")
    with c[3]: kpi_card("Correlation", num2(M.get("corr")), f"vs {res['bench']}" if has_b else "no benchmark")
    st.plotly_chart(growth_chart(rdates, M["equity"][1:],
                                 res["bench_eq"], res["bench"] if has_b else ""),
                    use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(donut_chart(tickers, w), use_container_width=True)
    with c2:
        st.plotly_chart(underwater_chart(rdates, M["equity"][1:]), use_container_width=True)

with tab_pf:
    st.plotly_chart(heatmap_fig(res["monthly"]), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(hist_chart(pr), use_container_width=True)
    with c2:
        st.plotly_chart(rolling_vol_chart(rdates, pr), use_container_width=True)
    pos_m = [r for _, r in res["monthly"] if r > 0]
    if has_b:
        bm_m = res["bench_monthly"]
        bm_pos = [r for _, r in bm_m if r > 0]
        bm_avg = np.mean([r for _, r in bm_m])
        bm_pos_rate = len(bm_pos) / len(bm_m)
        bm_best = max(bm_m, key=lambda x: x[1])
        bm_worst = min(bm_m, key=lambda x: x[1])
    c = st.columns(4)
    with c[0]: kpi_card("Avg monthly", pct(np.mean([r for _, r in res["monthly"]])),
                        f"{len(res['monthly'])} mo · Bench: {pct(bm_avg)}" if has_b else f"{len(res['monthly'])} months")
    with c[1]: kpi_card("Positive months", pct(len(pos_m) / len(res["monthly"]), 0),
                        f"Bench: {pct(bm_pos_rate, 0)}" if has_b else "share of months")
    with c[2]:
        ym, r = max(res["monthly"], key=lambda x: x[1])
        kpi_card("Best month", pct(r), ym + (f" · Bench {bm_best[0]}: {pct(bm_best[1])}" if has_b else ""), "pos")
    with c[3]:
        ym, r = min(res["monthly"], key=lambda x: x[1])
        kpi_card("Worst month", pct(r), ym + (f" · Bench {bm_worst[0]}: {pct(bm_worst[1])}" if has_b else ""), "neg")

with tab_rk:
    c = st.columns(4)
    with c[0]: kpi_card("VaR 95%", pct(M["var95"]), "daily · " + bmk("var95"), "neg")
    with c[1]: kpi_card("CVaR 95%", pct(M["cvar95"]), "daily ES · " + bmk("cvar95"), "neg")
    with c[2]: kpi_card("Downside dev", pct(M["down_dev"]), "ann. · " + bmk("down_dev"))
    with c[3]: kpi_card("Sortino", num2(M["sortino"]), f"rf {res['rf']*100:.1f}% · " + bmk("sortino", num2), cls(M["sortino"]))
    c = st.columns(4)
    with c[0]: kpi_card("Skewness", num2(M["skew"]), "daily · " + bmk("skew", num2))
    with c[1]: kpi_card("Kurtosis", num2(M["kurt"]), "excess · " + bmk("kurt", num2))
    with c[2]: kpi_card("Calmar", num2(M["calmar"]), "CAGR/|maxDD| · " + bmk("calmar", num2), cls(M["calmar"]))
    with c[3]: kpi_card("Win rate", pct(M["win_rate"], 0),
                        "pos. days · " + (f"Bench: {pct(BM['win_rate'], 0)}" if has_b else "no benchmark"))
    bench_tail = (f" · **Bench maxDD:** {pct(BM['max_dd'])}"
                  f" · **Bench best day:** {pct(BM['best_day'])}"
                  f" · **Bench worst day:** {pct(BM['worst_day'])}") if has_b else ""
    st.markdown(f"**Max drawdown:** {pct(M['max_dd'])} from {rdates[M['dd_peak']]} to {rdates[M['dd_trough']]} · "
                f"**Best day:** {pct(M['best_day'])} · **Worst day:** {pct(M['worst_day'])}" + bench_tail)
    if has_b:
        st.subheader("Benchmark-relative")
        rel = pd.DataFrame([
            ("Beta", num2(M["beta"])), ("Alpha (ann.)", pct(M["alpha"])),
            ("Correlation", num2(M["corr"])), ("R²", num2(M["r2"])),
            ("Tracking error", pct(M["te"])), ("Information ratio", num2(M["info"])),
            ("Treynor", pct(M["treynor"])), ("Upside capture", pct(M["up_cap"])),
            ("Downside capture", pct(M["dn_cap"])),
            (f"{res['bench']} CAGR", pct(M["bench_cagr"])),
            (f"{res['bench']} vol", pct(M["bench_vol"])),
        ], columns=["Metric", "Value"])
        st.dataframe(rel, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(rolling_beta_chart(rdates, pr, br), use_container_width=True)
        with c2:
            st.plotly_chart(corr_fig(tickers, res["rets_hold"]), use_container_width=True)
    rc = [h["risk_contrib"] for h in H]
    st.plotly_chart(risk_bar_fig(tickers, rc), use_container_width=True)

with tab_hd:
    rows = {
        "Ticker": list(tickers),
        "Weight": [pct(x) for x in w],
        "CAGR": [pct(h["cagr"]) for h in H],
        "Total return": [pct(h["total_ret"]) for h in H],
        "Volatility": [pct(h["vol"]) for h in H],
        "Sharpe": [num2(h["sharpe"]) for h in H],
        "Beta (vs portf.)": [num2(h["beta"]) for h in H],
        "Return contrib.": [pct(h["ret_contrib"]) for h in H],
        "Risk contrib.": [pct(h["risk_contrib"]) for h in H],
    }
    if has_b:  # benchmark twin row for direct comparison
        rows["Ticker"].append(f"{res['bench']} (bench)")
        rows["Weight"].append("—")
        rows["CAGR"].append(pct(BM["cagr"]))
        rows["Total return"].append(pct(BM["total_ret"]))
        rows["Volatility"].append(pct(BM["vol"]))
        rows["Sharpe"].append(num2(BM["sharpe"]))
        rows["Beta (vs portf.)"].append("1.00")
        rows["Return contrib."].append("—")
        rows["Risk contrib."].append("—")
    dfh = pd.DataFrame(rows)
    st.dataframe(dfh, use_container_width=True, hide_index=True)
    st.caption("Return contrib. ≈ weight × holding CAGR. Risk contrib. = Euler share of portfolio volatility (sums to 100%).")

with st.expander("Methodology & data notes"):
    st.markdown("""
- **Prices:** Yahoo Finance via `yfinance`, `auto_adjust=True` — splits and dividends are
  baked into closes, so returns are **total returns**. Series are inner-joined on common trading days.
- **CAGR** = (end/start)^(252/n) − 1. **Volatility** = daily σ × √252. **Sharpe** uses the risk-free rate above.
- **Sortino** = (CAGR − rf) / downside deviation, where downside deviation is the
  annualized root-mean-square of negative daily returns.
- **VaR/CVaR 95%** are historical (empirical quantile / tail mean of daily returns).
- **Beta** = cov(port, bench)/var(bench); **alpha** annualized; **R²** = corr²; **tracking error** = σ(port−bench) × √252;
  **information ratio** = mean(port−bench) × 252 / TE; **Treynor** = (CAGR − rf)/beta.
- **Up/down capture** = mean portfolio return on up/down benchmark days ÷ mean benchmark return on those days.
- **Risk contribution** is the Euler decomposition: wᵢ·(Σw)ᵢ / σ² — the share of portfolio variance/volatility
  attributable to each holding; sums to 100%.
- Every metric card shows its **benchmark twin** (same metric computed on the benchmark over the
  identical window), and the Holdings table ends with a benchmark row for direct comparison.
- **Sizing:** buy & hold lets position values compound (weights drift); rebalanced holds fixed target weights daily.
- Educational analysis, not investment advice.
""")

st.markdown("---")
st.markdown('<p class="src-note">Portfolio Analytics (Python edition) · data: Yahoo Finance · '
            'run with <span style="font-family:monospace">streamlit run app.py</span></p>',
            unsafe_allow_html=True)
