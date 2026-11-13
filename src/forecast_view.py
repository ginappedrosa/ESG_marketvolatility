"""Streamlit view for forward volatility and press coverage."""

from urllib.error import URLError

import pandas as pd
import streamlit as st

from press_context import GPR_PAGE, load_press_context, recent_headlines
from volatility_forecast import run_forecast

PILOT = {
    "S&P 500": ("United States", "America/New_York", 16),
    "EURO STOXX 50": ("Euro area", "Europe/Paris", 18),
    "IBEX 35": ("Spain", "Europe/Madrid", 18),
}


@st.cache_data(ttl=3600, show_spinner=False)
def press_snapshot():
    return load_press_context()


@st.cache_data(ttl=1800, show_spinner=False)
def headlines(region):
    return recent_headlines(region)


@st.cache_data(ttl=1800, show_spinner=False)
def evaluate(closes, press):
    return run_forecast(closes, press)


def show_forecast(index_symbols, load_closes):
    st.header("Volatility Forecast")
    st.caption("Pilot for S&P 500, EURO STOXX 50 and IBEX 35. Horizon: next five trading sessions.")
    index_name = st.selectbox("Index", list(PILOT))
    region, timezone, close_hour = PILOT[index_name]
    with st.spinner("Loading market data and evaluating the forecast"):
        try:
            closes = load_closes(index_symbols[index_name], "5y")
            local_time = pd.Timestamp.now(tz=timezone)
            if local_time.hour < close_hour:
                closes = closes.loc[closes.index < pd.Timestamp(local_time.date())]
            press, refreshed = press_snapshot()
            outcome = evaluate(closes, press)
        except (ValueError, KeyError, OSError) as exc:
            st.warning(f"Forecast unavailable: {exc}")
            outcome = None

    if outcome is not None:
        low, high = outcome["band"]
        left, right = st.columns(2)
        left.metric("Forecast daily volatility", f'{outcome["forecast"]:.2f}%')
        right.metric("Historical error band", f"{low:.2f}% – {high:.2f}%")
        st.caption(
            f'As of {outcome["as_of"]:%d %b %Y} · {outcome["chosen"]}. '
            "The band uses the middle 80% of errors from earlier validation windows."
        )
        if not refreshed:
            st.caption("Live GPR refresh unavailable. Using the archived releases.")
        if pd.notna(outcome["context_date"]):
            st.caption(f'Latest GPR input available from {outcome["context_date"]:%d %b %Y}.')
        st.subheader("Held-out backtest")
        test = outcome["test"]
        st.caption(
            f"{test.index[0]:%d %b %Y} to {test.index[-1]:%d %b %Y} · forecast dates. "
            "Lower MAE and RMSE are better. Stress MAE uses the upper 20% threshold from earlier data."
        )
        st.dataframe(outcome["scores"].round(3), width="stretch")
        st.line_chart(test[["Observed", outcome["chosen"]]])
        st.caption(
            f"Press features use published [Caldara–Iacoviello GPR releases]({GPR_PAGE}). "
            "This is a walk-forward test with a five-session gap before training cutoffs."
        )

    st.subheader("Recent international coverage")
    try:
        articles, source, time_label = headlines(region)
        st.caption(f"{source} · {time_label}. These individual headlines provide context for the charts.")
        if not articles:
            st.info("No matching headlines were returned.")
        else:
            st.dataframe(
                pd.DataFrame(articles), hide_index=True, width="stretch",
                column_config={"Article": st.column_config.LinkColumn("Article", display_text="Read")},
            )
    except (URLError, OSError, ValueError, KeyError):
        st.info("Recent headlines are temporarily unavailable.")
