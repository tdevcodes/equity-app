# app.py
# ---------------------------------------------------------
# Streamlit aplikace: Nahraj CSV -> Equity křivka + Drawdown + Trade Analysis (Weekday)
# ---------------------------------------------------------
# Funkce:
# - nahrání CSV (auto detekce oddělovače)
# - Equity křivka z balance nebo kumulativního P/L
# - Drawdown + základní metriky
# - Analýza obchodů podle dnů v týdnu: winrate, zisk, počet obchodů, PF, expectancy
# - Export grafů (PNG) a obohacených CSV

import io
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Equity & Trade Analysis", layout="wide")
st.title("📈 Equity & Trade Analysis z CSV")
st.write("Nahrajte CSV se stavem účtu nebo s obchody – aplikace vykreslí equity & drawdown a poskytne analýzu obchodů.")

# -------------- Pomocné funkce --------------

def read_csv_flexible(file) -> pd.DataFrame:
    """Načte CSV s autodetekcí oddělovače a normalizací čárek/teček u číselných polí."""
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

    df = normalize_numeric_strings(df)

    for col in df.columns:
        cl = col.lower()
        if any(k in cl for k in ["time", "date", "timestamp"]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            except Exception:
                pass
    return df


def normalize_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            series = out[col].astype(str)
            series = series.str.replace("\u00A0", " ", regex=False).str.strip()

            def to_float_safe(s: str):
                if not any(ch.isdigit() for ch in s):
                    return s
                # US: 1,234.56
                if pd.Series([s]).str.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$").iloc[0]:
                    s2 = s.replace(",", "")
                    try:
                        return float(s2)
                    except Exception:
                        return s
                # EU: 1.234,56
                if pd.Series([s]).str.match(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$").iloc[0]:
                    s2 = s.replace(".", "").replace(",", ".")
                    try:
                        return float(s2)
                    except Exception:
                        return s
                # Holé formáty
                if pd.Series([s]).str.match(r"^-?\d+(,\d+)?$").iloc[0]:
                    try:
                        return float(s.replace(",", "."))
                    except Exception:
                        return s
                if pd.Series([s]).str.match(r"^-?\d+(\.\d+)?$").iloc[0]:
                    try:
                        return float(s)
                    except Exception:
                        return s
                return s

            converted = series.apply(to_float_safe)
            try:
                out[col] = pd.to_numeric(converted, errors="ignore")
            except Exception:
                out[col] = converted
    return out


def pick_time_col(df: pd.DataFrame) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for key in [
        "closetime", "close time", "time", "date", "timestamp", "exit time", "exittime",
        "opentime", "open time", "entry time", "entrytime",
        "opendate", "closedate",
    ]:
        if key in lower:
            return lower[key]
    return None


def build_equity(df: pd.DataFrame, equity_source: str) -> Tuple[pd.Series, str]:
    lower = {c.lower(): c for c in df.columns}
    profit_col = lower.get("profitlossafterslippage") or lower.get("profitloss") or lower.get("pnl") or lower.get("pl")

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
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            col = num_cols[0]
            return pd.to_numeric(df[col], errors="coerce"), f"Equity = fallback '{col}'"
        return pd.Series(dtype=float), "Nebylo možné sestavit equity (žádná numerická data)."

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
    ax.set_title(title)
    ax.set_xlabel("Krok")
    ax.set_ylabel(ylabel)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# --------- Analýza obchodů (Dnů v týdnu) ---------

def combine_date_time(df: pd.DataFrame) -> pd.Series:
    """Zkusí vytvořit datetime z (OpenDate/EntryTime) nebo (CloseDate/ExitTime)."""
    lower = {c.lower(): c for c in df.columns}
    open_date = lower.get("opendate") or lower.get("entrydate") or lower.get("date")
    close_date = lower.get("closedate")
    entry_time = lower.get("entrytime") or lower.get("opentime")
    exit_time = lower.get("exittime") or lower.get("closetime")

    dt = None
    if open_date is not None:
        dt = pd.to_datetime(
            df[open_date].astype(str) + " " + df.get(entry_time, pd.Series([""]*len(df))).astype(str),
            errors="coerce",
        )
    if (dt is None or dt.isna().all()) and close_date is not None:
        dt = pd.to_datetime(
            df[close_date].astype(str) + " " + df.get(exit_time, pd.Series([""]*len(df))).astype(str),
            errors="coerce",
        )

    if dt is None:
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return df[col]
        return pd.Series([pd.NaT]*len(df))
    return dt


def weekday_metrics(df: pd.DataFrame, pnl_col: str) -> pd.DataFrame:
    dt = combine_date_time(df)
    pnl = pd.to_numeric(df[pnl_col], errors="coerce")
    temp = pd.DataFrame({"dt": dt, "pnl": pnl}).dropna(subset=["pnl"])  # dt může být NaT

    # 0=Mon ... 6=Sun; pokud nemáme datum, označíme -1
    if temp["dt"].notna().any():
        temp["weekday"] = temp["dt"].dt.dayofweek
    else:
        temp["weekday"] = -1

    # KLÍČOVÁ OPRAVA: agregujeme nad Series 'pnl'
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

    for col in ["Winrate", "AvgProfit", "MedianProfit", "TotalProfit", "ProfitFactor", "Expectancy"]:
        if col in out:
            out[col] = out[col].astype(float).round(4)
    return out


def bar_png(x, y, title: str, xlabel: str, ylabel: str) -> bytes:
    fig, ax = plt.subplots()
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# -------------- UI – Tabs --------------

tab1, tab2 = st.tabs(["Equity & Drawdown", "Trade Analysis (Dnů v týdnu)"])

with tab1:
    st.header("Equity & Drawdown")
    uploaded = st.file_uploader("Nahrajte CSV soubor (balance nebo P/L)", type=["csv"], key="eq_csv")

    if uploaded:
        df = read_csv_flexible(uploaded)
        time_col_name = pick_time_col(df)
        if time_col_name and pd.api.types.is_datetime64_any_dtype(df[time_col_name]):
            df = df.sort_values(time_col_name).reset_index(drop=True)

        st.success("CSV načteno.")
        with st.expander("Náhled dat", expanded=False):
            st.dataframe(df.head(50))

        choices = ["Auto"]
        for col in df.columns:
            if col.lower() in ("endingbalance", "openingbalance"):
                choices.append(col)
        choices.append("Cumulative ProfitLoss")

        equity_choice = st.selectbox("Zdroj equity", choices, index=0)

        equity, eq_label = build_equity(df, equity_choice)
        st.caption(eq_label)

        if equity is not None and not pd.Series(equity).dropna().empty:
            eq = pd.to_numeric(equity, errors="coerce")
            returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            roll_max = eq.cummax()
            drawdown = (eq - roll_max) / roll_max

            stats = compute_stats(eq, df[time_col_name] if time_col_name else None)
            st.subheader("📊 Základní metriky")
            st.dataframe(pd.DataFrame([stats]))

            st.subheader("📈 Grafy")
            col1, col2 = st.columns(2)
            with col1:
                png_eq = plot_series(eq, "Equity Curve", "Equity")
                st.image(png_eq, caption="Equity Curve", use_column_width=True)
                st.download_button("⬇️ Stáhnout Equity Curve (PNG)", data=png_eq, file_name="equity_curve.png", mime="image/png")
            with col2:
                png_dd = plot_series(drawdown.fillna(0.0), "Drawdown (relativní)", "Drawdown")
                st.image(png_dd, caption="Drawdown", use_column_width=True)
                st.download_button("⬇️ Stáhnout Drawdown (PNG)", data=png_dd, file_name="drawdown.png", mime="image/png")

            st.subheader("📥 Export obohaceného CSV")
            enriched = df.copy()
            enriched["EquityCurve"] = eq.values
            enriched["Return"] = returns.values
            enriched["LogReturn"] = np.log(eq / eq.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
            enriched["Drawdown"] = drawdown.values

            csv_buf = io.StringIO()
            enriched.to_csv(csv_buf, index=False)
            st.download_button("⬇️ Stáhnout equity_analysis.csv", data=csv_buf.getvalue(), file_name="equity_analysis.csv", mime="text/csv")
        else:
            st.error("Nepodařilo se sestavit equity křivku. Zkontrolujte, zda CSV obsahuje EndingBalance / OpeningBalance / ProfitLoss.")
    else:
        st.info("Nahrajte CSV soubor vlevo.")

with tab2:
    st.header("Trade Analysis – Dny v týdnu")
    trades_file = st.file_uploader("Nahrajte CSV s obchody (musí obsahovat P/L – ideálně 'ProfitLossAfterSlippage')", type=["csv"], key="trades_csv")

    if trades_file:
        df_t = read_csv_flexible(trades_file)

        # Určení sloupce s P/L
        lower = {c.lower(): c for c in df_t.columns}
        pnl_col = lower.get("profitlossafterslippage") or lower.get("profitloss") or lower.get("pnl") or lower.get("pl")
        if pnl_col is None:
            st.error("Nenašel jsem sloupec s P/L (zkuste 'ProfitLossAfterSlippage' nebo 'ProfitLoss').")
        else:
            st.caption(f"Použitý P/L sloupec: '{pnl_col}'")
            metrics = weekday_metrics(df_t, pnl_col)
            st.subheader("Souhrn podle dne v týdnu")
            st.dataframe(metrics, use_container_width=True)

            # Grafy: TotalProfit a Winrate podle Dnů
            if not metrics.empty:
                st.subheader("Grafy")
                c1, c2 = st.columns(2)
                with c1:
                    img_profit = bar_png(metrics["Weekday"], metrics["TotalProfit"], "Celkový zisk podle dne v týdnu", "Den", "Zisk")
                    st.image(img_profit, use_column_width=True)
                    st.download_button("⬇️ Stáhnout graf zisku (PNG)", data=img_profit, file_name="profit_by_weekday.png", mime="image/png")
                with c2:
                    img_win = bar_png(metrics["Weekday"], (metrics["Winrate"]*100.0).round(2), "Winrate podle dne v týdnu", "Den", "Winrate (%)")
                    st.image(img_win, use_column_width=True)
                    st.download_button("⬇️ Stáhnout graf winrate (PNG)", data=img_win, file_name="winrate_by_weekday.png", mime="image/png")

            # Export
            out_buf = io.StringIO()
            metrics.to_csv(out_buf, index=False)
            st.download_button("⬇️ Stáhnout weekday_metrics.csv", data=out_buf.getvalue(), file_name="weekday_metrics.csv", mime="text/csv")

            with st.expander("Náhled první tabulky souboru", expanded=False):
                st.dataframe(df_t.head(50))
    else:
        st.info("Nahrajte CSV s obchody.")