"""Five-session volatility forecasts using time-ordered market and press data."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HORIZON = 5
MIN_TRAIN = 252
TEST_BLOCK = 63
PRICE_FEATURES = ["vol5", "vol10", "vol22", "vol63", "return5", "drawdown"]
PRESS_FEATURES = ["gpr", "acts", "threats", "gpr_change", "release_age"]
VARIANTS = {
    "Ridge · prices": (PRICE_FEATURES, Ridge),
    "Gradient boosting · prices": (PRICE_FEATURES, HistGradientBoostingRegressor),
    "Ridge · prices + press": (PRICE_FEATURES + PRESS_FEATURES, Ridge),
    "Gradient boosting · prices + press": (
        PRICE_FEATURES + PRESS_FEATURES, HistGradientBoostingRegressor
    ),
}


def build_features(closes, press):
    returns = np.log(closes).diff()
    frame = pd.DataFrame(index=closes.index)
    for length in (5, 10, 22, 63):
        frame[f"vol{length}"] = np.sqrt(returns.pow(2).rolling(length).mean()) * 100
    frame["return5"] = returns.rolling(5).sum() * 100
    frame["drawdown"] = (closes / closes.rolling(22).max() - 1) * 100
    frame["target"] = frame["vol5"].shift(-HORIZON)
    frame["target_end"] = frame.index.to_series().shift(-HORIZON)
    if press.empty:
        frame[PRESS_FEATURES] = np.nan
        return frame

    context = press.sort_values("available_date").copy()
    context["gpr_change"] = context["gpr"].diff()
    aligned = pd.merge_asof(
        frame, context, left_index=True, right_on="available_date",
        tolerance=pd.Timedelta(days=45),
    )
    aligned["release_age"] = (aligned.index.to_series() - aligned["available_date"]).dt.days
    return aligned


def fit_model(train, columns, estimator):
    if estimator is Ridge:
        model = make_pipeline(StandardScaler(), Ridge(alpha=12))
    else:
        model = estimator(max_iter=100, max_leaf_nodes=7, min_samples_leaf=30,
                          l2_regularization=2, early_stopping=False, random_state=42)
    return model.fit(train[columns], np.log(train["target"].clip(lower=0.001)))


def estimate(model, rows, columns):
    return np.exp(model.predict(rows[columns]))


def score_table(predictions, target, stress_threshold):
    errors = predictions.sub(target, axis=0)
    scores = pd.DataFrame({
        "MAE (pp)": errors.abs().mean(),
        "RMSE (pp)": errors.pow(2).mean().pow(0.5),
        "Stress MAE (pp)": errors.loc[target >= stress_threshold].abs().mean(),
        "R²": 1 - errors.pow(2).sum() / target.sub(target.mean()).pow(2).sum(),
    })
    return scores.sort_values("MAE (pp)")


def run_forecast(closes, press):
    with threadpool_limits(limits=1):
        return _run_forecast(closes, press)


def _run_forecast(closes, press):
    frame = build_features(closes, press).dropna(subset=PRICE_FEATURES)
    observations = frame.dropna(subset=["target"])
    price_variants = [name for name, (cols, _) in VARIANTS.items() if cols == PRICE_FEATURES]
    variants = price_variants
    eligible = observations.dropna(subset=PRESS_FEATURES)
    if len(eligible) >= MIN_TRAIN + 3 * TEST_BLOCK + HORIZON:
        variants = list(VARIANTS)
        observations = eligible
    if len(observations) < MIN_TRAIN + 3 * TEST_BLOCK + HORIZON:
        raise ValueError("At least about two years of overlapping, usable sessions are required.")

    # Reserve the most recent year or so for rolling evaluation. Its final block
    # is held out from selection. Training labels must end before each block starts.
    start = max(MIN_TRAIN + HORIZON, len(observations) - 5 * TEST_BLOCK)
    starts = list(range(start, len(observations) - TEST_BLOCK + 1, TEST_BLOCK))
    if starts[-1] != len(observations) - TEST_BLOCK:
        starts.append(len(observations) - TEST_BLOCK)
    predictions = []
    for index, position in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(observations)
        test = observations.iloc[position:end]
        train = observations.loc[observations["target_end"] < test.index[0]]
        if len(train) < MIN_TRAIN:
            continue
        forecasts = {"Recent volatility · baseline": test["vol5"]}
        for name in variants:
            columns, estimator = VARIANTS[name]
            forecasts[name] = estimate(fit_model(train, columns, estimator), test, columns)
        predictions.append(pd.DataFrame(forecasts, index=test.index))
    if len(predictions) < 2:
        raise ValueError("Insufficient sessions for model selection and a held-out test.")

    validation = pd.concat(predictions[:-1])
    holdout = predictions[-1]
    actual = observations["target"]
    # Exclude selection forecasts whose outcome overlaps the held-out window.
    validation = validation.loc[observations.loc[validation.index, "target_end"] < holdout.index[0]]
    stress_threshold = actual.loc[observations["target_end"] < holdout.index[0]].quantile(0.8)
    selection = score_table(validation, actual.loc[validation.index], stress_threshold)
    scores = score_table(holdout, actual.loc[holdout.index], stress_threshold)
    latest = frame.iloc[[-1]]
    available = [name for name in selection.index
                 if name not in VARIANTS or latest[VARIANTS[name][0]].notna().all(axis=None)]
    chosen = available[0]
    forecast = latest["vol5"].iat[0]
    if chosen in VARIANTS:
        columns, estimator = VARIANTS[chosen]
        train = observations.dropna(subset=columns)
        if columns == PRICE_FEATURES:
            train = frame.dropna(subset=["target"])
        forecast = float(estimate(fit_model(train, columns, estimator), latest, columns)[0])
    residuals = (actual.loc[validation.index] - validation[chosen]).dropna()
    # Use non-overlapping five-session blocks for the empirical error band.
    low, high = residuals.iloc[::HORIZON].quantile([0.1, 0.9])
    return {
        "forecast": forecast, "band": (max(0, forecast + low), max(0, forecast + high)),
        "chosen": chosen, "scores": scores, "test": holdout.assign(Observed=actual),
        "as_of": latest.index[0], "context_date": latest.get("available_date", pd.Series([pd.NaT])).iat[0],
    }
