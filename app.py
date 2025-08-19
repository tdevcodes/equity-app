# app.py
# ---------------------------------------------------------
# Streamlit aplikace: Nahraj CSV -> Equity křivka + Drawdown
# ---------------------------------------------------------
# Funkce:
# - nahrání CSV (auto detekce oddělovače)
# - volba sloupce pro equity (EndingBalance / OpeningBalance / kumulativní ProfitLoss)
# - normalizace desetinných čárek/teček
# - výpis základních metrik
# - graf Equity Curve a Drawdown (PNG ke stažení)
# - export obohaceného CSV

import io
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Equity & Drawdown z CSV", layout="wide")
st.title("📈 Equity & Drawdown z CSV")
st.write("Nahrajte CSV soubor se sloupci jako **OpeningBalance**, **EndingBalance** nebo **ProfitLoss**. Aplikace vykreslí equity křivku, drawdown a připraví ke stažení obohacený CSV.")

# -------------- Pomocné funkce --------------

def read_csv_flexible(file) -> pd.DataFrame:
    """Načte CSV s autodetekcí oddělovače a normalizací čárek/teček u číselných polí."""
    data = file.read()
    # preview jako text kvůli heuristice
    sample = data[:4096].decode("utf-8", errors="ignore")

    # Určení potenciálního oddělovače
    sep = None
    if sample.count(";") > sample.count(","):
        sep = ";"
    # Pokud řádky vypadají jako CSV s čárkou
    elif sample.count(",") > 0:
        sep = ","

    # Načtení do DataFrame
    if sep:
        df = pd.read_csv(io.BytesIO(data), sep=sep)
    else:
        df = pd.read_csv(io.BytesIO(data), sep=None, engine="python")

    # Normalizace číselných stringů (1.234,56 | 1,234.56 | 1234,56 | 1234.56)
    df = normalize_numeric_strings(df)

    # Pokus o parse datumových sloupců
    for col in df.columns:
        cl = col.lower()
        if any(k in cl for k in ["time", "date", "timestamp"]):
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
            except Exception:
                pass
    return df


def normalize_numeric_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Pro každý sloupec typu object se pokusí převést textové číselné formáty na float.
    Zachová nečíselné texty beze změny."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            series = out[col].astype(str)
            # Odstranění NBSP a trim
            series = series.str.replace("\u00A0", " ", regex=False).str.strip()

            def to_float_safe(s: str):
                # když nic číselného, vrať původní
                if not any(ch.isdigit() for ch in s):
                    return s
                # US: 1,234.56 -> 1234.56
                if pd.Series([s]).str.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$").iloc[0]:
                    s2 = s.replace(",", "")
                    try:
                        return float(s2)
                    except Exception:
                        return s
                # EU: 1.234,56 -> 1234.56
                if pd.Series([s]).str.match(r"^-?\d{1,3}(\.\d{3})+(,\d+)?$").iloc[0]:
                    s2 = s.replace(".", "").replace(",", ".")
                    try:
                        return float(s2)
                    except Exception:
                        return s
                # Holá desetinná tečka / čárka
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
            # Pokus o převod na numeric – co nešlo, zůstane jako object
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
    ]:
        if key in lower:
            return lower[key]
    return None


def build_equity(df: pd.DataFrame, equity_source: str) -> Tuple[pd.Series, str]:
    """Vrátí equity sérii dle zvoleného zdroje."""
    lower = {c.lower(): c for c in df.columns}
    balance_cols = [c for c in df.columns if c.lower() in ["openingbalance", "endingbalance"]]
    profit_col = lower.get("profitloss")

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
        return pd.Series(dtype=float), "Sloupec 'ProfitLoss' nebyl nalezen."

    return pd.Series(dtype=float), "Neznámý zdroj equity."


def compute_stats(eq: pd.Series, time_col: Optional[pd.Series]):
    eq = pd.to_numeric(eq, errors="coerce")
    eq = eq.dropna()
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

    # zkus odhad annualizačního faktoru
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

# -------------- UI --------------

uploaded = st.file_uploader("Nahrajte CSV soubor", type=["csv"]) 

if uploaded:
    df = read_csv_flexible(uploaded)

    # Seřazení podle času, pokud máme časový sloupec
    time_col_name = pick_time_col(df)
    if time_col_name and pd.api.types.is_datetime64_any_dtype(df[time_col_name]):
        df = df.sort_values(time_col_name).reset_index(drop=True)

    st.success("CSV načteno.")
    with st.expander("Náhled dat", expanded=False):
        st.dataframe(df.head(50))

    # Volba zdroje equity
    choices = ["Auto"]
    for col in df.columns:
        if col.lower() in ("endingbalance", "openingbalance"):
            choices.append(col)
    choices.append("Cumulative ProfitLoss")

    equity_choice = st.selectbox("Zdroj equity", choices, index=0)

    equity, eq_label = build_equity(df, equity_choice)
    st.caption(eq_label)

    if equity is not None and not equity.dropna().empty:
        # výpočty
        eq = pd.to_numeric(equity, errors="coerce")
        returns = eq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        roll_max = eq.cummax()
        drawdown = (eq - roll_max) / roll_max

        # metriky
        stats = compute_stats(eq, df[time_col_name] if time_col_name else None)
        st.subheader("📊 Základní metriky")
        st.dataframe(pd.DataFrame([stats]))

        # grafy
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

        # Obohacený CSV
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
