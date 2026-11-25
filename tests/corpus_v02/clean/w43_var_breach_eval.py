# expect: none
import numpy as np
import pandas as pd


def evaluate(df, cost=0.0005):
    df["net"] = df["position"].shift(1) * df["close"].pct_change() - cost * df["position"].diff().abs()
    var95 = df["net"].quantile(0.05)
    df["breach"] = df["net"] < var95
    return {"sharpe": df["net"].mean() / df["net"].std() * np.sqrt(252), "breaches": df["breach"].mean()}
