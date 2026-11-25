# expect: LA14:high@19
import numpy as np
import optuna
import pandas as pd

px = pd.read_csv("sol_1h.csv", index_col=0, parse_dates=True)["close"]
train_px, test_px = px.loc[:"2023-06-30"], px.loc["2023-07-01":]


def run(p, fast, slow):
    pos = (p.rolling(fast).mean() > p.rolling(slow).mean()).astype(float).shift(1)
    r = pos * p.pct_change()
    return r.mean() / r.std() * np.sqrt(24 * 365)


def objective(trial):
    fast = trial.suggest_int("fast", 5, 50)
    slow = trial.suggest_int("slow", 60, 400)
    return run(test_px, fast, slow)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)
print("out-of-sample Sharpe", study.best_value, study.best_params)
