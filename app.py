# app.py
# ---------------------------------------------------------
# Streamlit: Equity & Drawdown  +  Trade Analysis (weekday)
# ---------------------------------------------------------

import io
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Equity & Trade Analysis", layout="wide")
st.title("📊 Trading CSV Analyzer")

# ---------- Helpers ----------
def read_csv_flexible(file) -> pd.DataFrame:
    data = file.read()
    sample = data[:4096].decode("utf-8", errors="ignore")
    sep = ";" if sample.count(";") > sample.count(",") else ("," if sample.count(",") > 0 else None)
    df = pd.read_csv(io.BytesIO(data), sep=sep) if sep else pd.read_csv(io.BytesIO(data), sep=None, engine="python")
    return normalize_numeric_strings(df)

def normalize_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            s = out[col].astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
            def to_float_safe(x: str):
                if not any(ch.isdigit() for ch in x):
                    return x
                try:
                    return float(x.replace(" ", "").replace(",", "."))
                except Exception:
                    return x
            out[col] = s.apply(to_float_safe)
    return out

def pick_time_col(df: pd.DataFrame) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for key in ["closetime","close time","time","date","timestamp","exit time","exittime",
                "opentime","open time","entry time","entrytime","opendate","closedate"]:
        if key in lower:
            return lower[key]
    return None

# ---- původní ověřená logika pro Equity (beze změn) ----
def build_equity(df: pd.DataFrame, equity_source: str) -> Tuple[pd.Series, str]:
    lower = {c.lower(): c for c in df.columns}
    profit_col = lower.get("profitloss") or lower.get("pnl") or lower.get("pl")

    if equity_source == "Auto":
        if "endingbalance" in lower:
            col = lower["endingbalance"]
            return pd.to_numeric(df[col], errors="coerce"), f"Equity = '{col}'"
        if "openingbalance" in lower:
            col = lower["openingbalance"]
            return pd.to_numeric(df[col], errors="coerce"), f"Equity = '{col}'"
        if profit_col:
            eq = pd.to_numeric(df[profit_col], errors="coerce").fillna(0).cumsum()
            return eq, f"Equity = kumulativní '{profit_col}' (start 0)"
        # fallback: první numerický sloupec
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            col = num_cols[0]
            return pd.to_numeric(df[col], errors="coerce"), f"Equity = fallback '{col}'"
        return pd.Series(dtype=float), "Nebylo možné sestavit equity (žádná numerická data)."

    # Explicitní volby
    if equity_source in df.columns:
        return pd.to_numeric(df[equity_source], errors="coerce"), f"Equity = '{equity_source}'"

    if equity_source == "Cumulative ProfitLoss":
        if profit_col:
            eq = pd.to_numeric(df[profit_col], errors="coerce").fillna(0).cumsum()
            return eq, f"Equity = kumulativní '{profit_col}'"
        return pd.Series(dtype=float), "Sloupec s P/L nenalezen."

    return pd.Series(dtype=float), "Neznámý zdroj equity."

def compute_stats(eq: pd.Series, time_col: Optional[pd.Series]):
    eq = pd.to_numeric(eq, errors="coerce").dropna()
    if eq.empty or len(eq) < 2:
        return {
            "Points": len(eq),
            "First equity": np.nan,
            "Last equity": np.nan,
            "Total return": np.nan,
            "Max drawdown": np.nan,
            "Sharpe (approx)": np.nan,
        }

    returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    roll_max = eq.cummax()
    drawdown = (eq - roll_max) / roll_max

    total_return = (eq.iloc[-1] / eq.iloc[0] - 1.0) if eq.iloc[0] != 0 else np.nan
    max_dd = float(drawdown.min()) if not drawdown.empty else np.nan

    # přibližný annualizační faktor
    periods_per_year = 252.0
    ann_factor = math.sqrt(periods_per_year)
    if time_col is not None and pd.api.types.is_datetime64_any_dtype(time_col):
        if len(time_col.dropna()) >= 2:
            days = (time_col.iloc[-1] - time_col.iloc[0]).total_seconds() / (3600 * 24)
            steps_per_day = len(eq) / max(days, 1)
            base = max(steps_per_day * 252.0, 1.0)
            ann_factor = math.sqrt(base)

    sharpe = (returns.mean() / (returns.std() + 1e-12)) * ann_factor if returns.std() > 0 else np.nan

    return {
        "Points": int(len(eq)),
        "First equity": float(eq.iloc[0]),
        "Last equity": float(eq.iloc[-1]),
        "Total return": float(total_return),
        "Max drawdown": float(max_dd),
        "Sharpe (approx)": float(sharpe) if not math.isnan(sharpe) else np.nan,
    }

def plot_series(series: pd.Series, title: str, ylabel: str) -> bytes:
    fig, ax = plt.subplots()
    ax.plot(series.values)
    ax.set_title(title); ax.set_xlabel("Krok"); ax.set_ylabel(ylabel)
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig); buf.seek(0)
    return buf.read()

# ---------- Trade Analysis helpers ----------
def combine_date_time_manual(df: pd.DataFrame, date_col: str, time_col: Optional[str], dayfirst: bool, hour_shift: int = 0) -> pd.Series:
    if time_col and time_col != "(žádný)":
        ts = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str),
                            errors="coerce", dayfirst=dayfirst)
    else:
        ts = pd.to_datetime(df[date_col], errors="coerce", dayfirst=dayfirst)
    if hour_shift:
        ts = ts + pd.to_timedelta(hour_shift, unit="h")
    return ts

def weekday_metrics_from(df: pd.DataFrame, pnl_col: str, ts: pd.Series) -> pd.DataFrame:
    pnl = pd.to_numeric(df[pnl_col], errors="coerce")
    temp = pd.DataFrame({"dt": ts, "pnl": pnl}).dropna(subset=["pnl"])

    if temp["dt"].notna().any():
        temp["weekday"] = temp["dt"].dt.dayofweek
    else:
        temp["weekday"] = -1

    grp = temp.groupby("weekday", dropna=False)["pnl"]

    def profit_factor(s: pd.Series) -> float:
        pos = s[s > 0].sum()
        neg = -s[s < 0].sum()
        return float(pos / neg) if neg > 0 else np.nan

    out = grp.agg(
        Trades="count",
        Wins=lambda s: int((s > 0).sum()),
        Losses=lambda s: int((s < 0).sum()),
        Winrate=lambda s: float((s > 0).mean()) if len(s) else np.nan,
        TotalProfit="sum",
        AvgProfit="mean",
        MedianProfit="median",
        ProfitFactor=profit_factor,
        Expectancy="mean",
    ).reset_index()

    name_map = {-1: "Unknown", 0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    out["Weekday"] = out["weekday"].map(name_map)
    out = out.drop(columns=["weekday"]).sort_values("Weekday")

    for col in ["Winrate","AvgProfit","MedianProfit","TotalProfit","ProfitFactor","Expectancy"]:
        if col in out:
            out[col] = out[col].astype(float).round(4)
    return out

def bar_png(x, y, title: str, xlabel: str, ylabel: str) -> bytes:
    # bezpečné typy pro Matplotlib (None -> "Unknown")
    x_s = pd.Series(x).fillna("Unknown").astype(str).tolist()
    y_s = pd.to_numeric(pd.Series(y), errors="coerce").fillna(0).tolist()
    fig, ax = plt.subplots()
    ax.bar(x_s, y_s)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig); buf.seek(0)
    return buf.read()

# ---------- UI: dvě viditelné záložky (každá s vlastním uploaderem) ----------
tab1, tab2 = st.tabs(["Equity & Drawdown", "Trade Analysis (dny v týdnu)"])

# ===== Tab 1: Equity & Drawdown (původní, beze změn) =====
with tab1:
    st.subheader("📈 Equity & Drawdown")
    up1 = st.file_uploader("Nahrajte CSV se stavem účtu / P&L", type=["csv"], key="eq_csv")
    if up1:
        df = read_csv_flexible(up1)

        # seřazení podle času (pokud máme časový sloupec)
        time_col_name = pick_time_col(df)
        if time_col_name and pd.api.types.is_datetime64_any_dtype(df[time_col_name]):
            df = df.sort_values(time_col_name).reset_index(drop=True)

        with st.expander("Náhled dat"):
            st.dataframe(df.head(50), use_container_width=True)

        choices = ["Auto"] + [c for c in df.columns if c.lower() in ("endingbalance","openingbalance")] + ["Cumulative ProfitLoss"]
        equity_choice = st.selectbox("Zdroj equity", choices, index=0)

        equity, eq_label = build_equity(df, equity_choice)
        st.caption(eq_label)

        if not pd.Series(equity).dropna().empty:
            eq = pd.to_numeric(equity, errors="coerce")
            returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            roll_max = eq.cummax()
            drawdown = (eq - roll_max) / roll_max

            stats = compute_stats(eq, df[time_col_name] if time_col_name else None)
            st.dataframe(pd.DataFrame([stats]), use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                png_eq = plot_series(eq, "Equity Curve", "Equity")
                st.image(png_eq, caption="Equity Curve", use_container_width=True)
                st.download_button("⬇️ PNG – Equity Curve", data=png_eq, file_name="equity_curve.png", mime="image/png")
            with c2:
                png_dd = plot_series(drawdown.fillna(0.0), "Drawdown (relativní)", "DD")
                st.image(png_dd, caption="Drawdown", use_container_width=True)
                st.download_button("⬇️ PNG – Drawdown", data=png_dd, file_name="drawdown.png", mime="image/png")

            st.subheader("📥 Export obohaceného CSV")
            enriched = df.copy()
            enriched["EquityCurve"] = eq.values
            enriched["Return"] = returns.values
            enriched["LogReturn"] = np.log(eq/eq.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
            enriched["Drawdown"] = drawdown.values

            csv_buf = io.StringIO()
            enriched.to_csv(csv_buf, index=False)
            st.download_button("⬇️ Stáhnout equity_analysis.csv", data=csv_buf.getvalue(), file_name="equity_analysis.csv", mime="text/csv")
        else:
            st.warning("Nepodařilo se sestavit equity křivku. Zkontrolujte EndingBalance / OpeningBalance / ProfitLoss.")

# ===== Tab 2: Trade Analysis (nová) =====
with tab2:
    st.subheader("📊 Analýza obchodů podle dnů v týdnu")
    up2 = st.file_uploader("Nahrajte CSV s obchody (musí obsahovat P/L)", type=["csv"], key="trades_csv")
    if up2:
        df_t = read_csv_flexible(up2)

        # 2) Přesné texty u rolovátek
        lower = {c.lower(): c for c in df_t.columns}
        auto_pl = lower.get("profitlossafterslippage") or lower.get("profitloss") or lower.get("pnl") or lower.get("pl")
        pl_choice = st.selectbox("Sloupec s P/L (ProfitLossAfterSlippage)",
                                 [auto_pl] + [c for c in df_t.columns if c != auto_pl] if auto_pl else list(df_t.columns))

        options = ["(žádný)"] + df_t.columns.tolist()
        date_col = st.selectbox("Sloupec s datem (OpenDate) ", options)
        time_col = st.selectbox("Sloupec s časem (OpenTime)", options)
        dayfirst = st.checkbox("Použít dayfirst (DD/MM vs MM/DD)", value=True)
        shift = st.number_input("Posun času (hodiny, např. +6 = posun do US/Eastern)", value=0, step=1, format="%d")

        if not pl_choice:
            st.error("Vyberte sloupec s P/L.")
        elif date_col == "(žádný)":
            st.warning("Vyberte sloupec s datem.")
        else:
            ts = combine_date_time_manual(df_t, date_col, None if time_col == "(žádný)" else time_col, dayfirst, int(shift))
            metrics = weekday_metrics_from(df_t, pl_choice, ts)

            # očista Weekday, aby grafy nikdy nespadly
            metrics["Weekday"] = metrics["Weekday"].fillna("Unknown").astype(str)

            st.dataframe(metrics, use_container_width=True)

            if not metrics.empty:
                c1, c2 = st.columns(2)
                with c1:
                    img_profit = bar_png(metrics["Weekday"], metrics["TotalProfit"], "Celkový zisk podle dne v týdnu", "Den", "Zisk")
                    st.image(img_profit, use_container_width=True)
                    st.download_button("⬇️ PNG – zisk podle dne", img_profit, "profit_by_weekday.png", "image/png")
                with c2:
                    img_win = bar_png(metrics["Weekday"], (metrics["Winrate"]*100.0).round(2), "Winrate podle dne v týdnu", "Den", "Winrate (%)")
                    st.image(img_win, use_container_width=True)
                    st.download_button("⬇️ PNG – winrate podle dne", img_win, "winrate_by_weekday.png", "image/png")

                out_buf = io.StringIO()
                metrics.to_csv(out_buf, index=False)
                st.download_button("⬇️ Stáhnout weekday_metrics.csv", out_buf.getvalue(), "weekday_metrics.csv", "text/csv")
