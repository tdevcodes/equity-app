# app.py
# ---------------------------------------------------------
# Streamlit aplikace: Equity/Drawdown + Trade Analysis
# ---------------------------------------------------------

import io
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Equity & Trade Analysis", layout="wide")

# ---------------------------------------------------------
# Pomocné funkce
# ---------------------------------------------------------

def read_csv_flexible(file) -> pd.DataFrame:
    data = file.read()
    sample = data[:4096].decode("utf-8", errors="ignore")
    sep = None
    if sample.count(";") > sample.count(","):
        sep = ";"
    elif sample.count(",") > 0:
        sep = ","
    if sep:
        df = pd.read_csv(io.BytesIO(data), sep=sep)
    else:
        df = pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    return normalize_numeric_strings(df)

def normalize_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            series = out[col].astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
            def to_float_safe(s: str):
                if not any(ch.isdigit() for ch in s):
                    return s
                try:
                    return float(s.replace(",", ".").replace(" ", ""))
                except Exception:
                    return s
            out[col] = series.apply(to_float_safe)
    return out

def build_equity(df: pd.DataFrame, equity_source: str) -> Tuple[pd.Series, str]:
    lower = {c.lower(): c for c in df.columns}
    profit_col = lower.get("profitloss")
    if equity_source == "Auto":
        if "endingbalance" in lower:
            return pd.to_numeric(df[lower["endingbalance"]], errors="coerce"), "Equity = EndingBalance"
        if "openingbalance" in lower:
            return pd.to_numeric(df[lower["openingbalance"]], errors="coerce"), "Equity = OpeningBalance"
        if profit_col:
            return pd.to_numeric(df[profit_col], errors="coerce").fillna(0).cumsum(), "Equity = Cumulative ProfitLoss"
    if equity_source in df.columns:
        return pd.to_numeric(df[equity_source], errors="coerce"), f"Equity = {equity_source}"
    if equity_source == "Cumulative ProfitLoss" and profit_col:
        return pd.to_numeric(df[profit_col], errors="coerce").fillna(0).cumsum(), "Equity = Cumulative ProfitLoss"
    return pd.Series(dtype=float), "Nebylo možné sestavit equity."

def compute_stats(eq: pd.Series):
    eq = pd.to_numeric(eq, errors="coerce").dropna()
    if len(eq) < 2:
        return {"Points": len(eq), "First": np.nan, "Last": np.nan, "Return": np.nan, "MaxDD": np.nan}
    returns = eq.pct_change().fillna(0)
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    return {
        "Points": len(eq),
        "First": float(eq.iloc[0]),
        "Last": float(eq.iloc[-1]),
        "Return": float(eq.iloc[-1]/eq.iloc[0] - 1),
        "MaxDD": float(dd.min())
    }

def plot_series(series: pd.Series, title: str, ylabel: str) -> bytes:
    fig, ax = plt.subplots()
    ax.plot(series.values)
    ax.set_title(title)
    ax.set_xlabel("Krok")
    ax.set_ylabel(ylabel)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ---------------------------------------------------------
# Aplikace
# ---------------------------------------------------------

st.title("📊 Trading CSV Analyzer")

uploaded = st.file_uploader("Nahrajte CSV soubor", type=["csv"])
if not uploaded:
    st.info("⬆️ Nahrajte CSV soubor vlevo nahoře")
    st.stop()

df = read_csv_flexible(uploaded)
st.success("CSV načteno")

tabs = st.tabs(["Equity & Drawdown", "Trade Analysis"])

# ---------------------------------------------------------
# Equity & Drawdown
# ---------------------------------------------------------
with tabs[0]:
    st.header("📈 Equity & Drawdown")
    with st.expander("Náhled dat"):
        st.dataframe(df.head(50))

    choices = ["Auto"]
    for col in df.columns:
        if col.lower() in ("endingbalance", "openingbalance"):
            choices.append(col)
    choices.append("Cumulative ProfitLoss")
    equity_choice = st.selectbox("Zdroj equity", choices, index=0)
    equity, eq_label = build_equity(df, equity_choice)
    st.caption(eq_label)

    if not equity.empty:
        stats = compute_stats(equity)
        st.dataframe(pd.DataFrame([stats]))
        col1, col2 = st.columns(2)
        with col1:
            png_eq = plot_series(equity, "Equity Curve", "Equity")
            st.image(png_eq, caption="Equity Curve", use_container_width=True)
            st.download_button("⬇️ PNG Equity", png_eq, "equity.png")
        with col2:
            roll_max = equity.cummax()
            dd = (equity-roll_max)/roll_max
            png_dd = plot_series(dd, "Drawdown", "DD")
            st.image(png_dd, caption="Drawdown", use_container_width=True)
            st.download_button("⬇️ PNG DD", png_dd, "dd.png")
    else:
        st.error("Nepodařilo se sestavit equity křivku.")

# ---------------------------------------------------------
# Trade Analysis
# ---------------------------------------------------------
with tabs[1]:
    st.header("📊 Trade Analysis podle dne v týdnu")

    # výběr sloupců pro čas
    col_options = ["(žádný)"] + df.columns.tolist()
    date_col = st.selectbox("Sloupec s DATEM", col_options)
    time_col = st.selectbox("Sloupec s ČASEM (volitelné)", col_options)
    dayfirst = st.checkbox("Použít dayfirst (DD/MM vs MM/DD)", value=True)

    ts = None
    if date_col != "(žádný)":
        if time_col != "(žádný)":
            ts = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce", dayfirst=dayfirst)
        else:
            ts = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst)

    if ts is None or ts.dropna().empty:
        st.warning("Nepodařilo se složit časové razítko. Zkontrolujte sloupce a formát.")
        st.stop()

    df["_weekday"] = ts.dt.day_name(locale="en_US")  # Mon, Tue…
    if "ProfitLoss" not in df.columns:
        st.error("Chybí sloupec ProfitLoss pro výpočet.")
        st.stop()

    df["_pl"] = pd.to_numeric(df["ProfitLoss"], errors="coerce").fillna(0)
    grouped = df.groupby("_weekday")["_pl"].agg(["count", "sum", "mean"])
    st.dataframe(grouped)

    fig1, ax1 = plt.subplots()
    grouped["sum"].plot(kind="bar", ax=ax1)
    ax1.set_title("Celkový zisk podle dne")
    ax1.set_ylabel("Zisk")
    st.pyplot(fig1, use_container_width=True)

    fig2, ax2 = plt.subplots()
    winrate = (df["_pl"] > 0).groupby(df["_weekday"]).mean()*100
    winrate.plot(kind="bar", ax=ax2)
    ax2.set_title("Winrate podle dne")
    ax2.set_ylabel("Winrate %")
    st.pyplot(fig2, use_container_width=True)
