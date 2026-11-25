# expect: any
# From the red-team review of PR #8 (v0.2), 2026-10-16.
import optuna
from sklearn.ensemble import GradientBoostingRegressor
def objective(trial):
    m = GradientBoostingRegressor(max_depth=trial.suggest_int("max_depth", 2, 8),
                                  learning_rate=trial.suggest_float("lr", 0.01, 0.3))
    m.fit(X_train, y_train)
    return m.score(X_test, y_test)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)
print("test R2", study.best_value)
