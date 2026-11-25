"""Catalog of every check the auditor runs: what it is, why it inflates results, how to fix it.

`explain(check_id)` serves these entries. Each code-lint finding takes its default `why` and
`fix` text from here, so the explanation a user reads is the same everywhere.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

SCORE_CAVEAT = (
    "A score of 100 means none of our listed patterns matched, not that the code is free of leaks. "
    "Static checks miss many leaks."
)
NO_BENCHMARK_REASON = "No benchmark supplied: market exposure can't be separated from skill."

DISCLAIMER = (
    "This is an automated statistical and code review of a hypothetical backtest. It is general information, "
    "not investment advice, not tailored to anyone, and does not predict future results. The checks can miss "
    "problems and can flag correct code. Nothing But Alpha is not a registered investment adviser."
)

# Shared references (full citations, reused by several checks).
REF_DSR = (
    "Bailey, D. H. & López de Prado, M. (2014). The Deflated Sharpe Ratio: Correcting for "
    "Selection Bias, Backtest Overfitting and Non-Normality. Journal of Portfolio Management 40(5), 94-107."
)
REF_FRONTIER = (
    "Bailey, D. H. & López de Prado, M. (2012). The Sharpe Ratio Efficient Frontier. "
    "Journal of Risk 15(2), 3-44."
)
REF_LO = "Lo, A. W. (2002). The Statistics of Sharpe Ratios. Financial Analysts Journal 58(4), 36-52."
REF_NW = (
    "Newey, W. K. & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and "
    "Autocorrelation Consistent Covariance Matrix. Econometrica 55(3), 703-708."
)
REF_AFML = "López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley. Ch. 7 (cross-validation), Ch. 11-12 (backtesting)."
REF_PSEUDO = (
    "Bailey, D. H., Borwein, J., López de Prado, M. & Zhu, Q. J. (2014). Pseudo-Mathematics and "
    "Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance. "
    "Notices of the AMS 61(5), 458-471."
)
REF_HLZ = "Harvey, C. R., Liu, Y. & Zhu, H. (2016). ... and the Cross-Section of Expected Returns. Review of Financial Studies 29(1), 5-68."
REF_WHITE = "White, H. (2000). A Reality Check for Data Snooping. Econometrica 68(5), 1097-1126."
REF_LEAK = (
    "Kaufman, S., Rosset, S., Perlich, C. & Stitelman, O. (2012). Leakage in Data Mining: Formulation, "
    "Detection, and Avoidance. ACM Transactions on Knowledge Discovery from Data 6(4)."
)
REF_SURV = (
    "Brown, S. J., Goetzmann, W., Ibbotson, R. G. & Ross, S. A. (1992). Survivorship Bias in "
    "Performance Studies. Review of Financial Studies 5(4), 553-580."
)
REF_SURV2 = "Elton, E. J., Gruber, M. J. & Blake, C. R. (1996). Survivorship Bias and Mutual Fund Performance. Review of Financial Studies 9(4), 1097-1120."
REF_COSTS = "Novy-Marx, R. & Velikov, M. (2016). A Taxonomy of Anomalies and Their Trading Costs. Review of Financial Studies 29(1), 104-147."
REF_SKLEARN_TS = "scikit-learn user guide, 'Cross-validation of time series data' (TimeSeriesSplit)."
REF_PANDAS_RESAMPLE = "pandas documentation, DataFrame.resample: parameters `closed` and `label`."


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    category: str  # lookahead | survivorship | data | costs | overfitting | statistics | stability | concentration
    what: str
    why_it_inflates: str
    how_to_fix: str
    references: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["references"] = list(self.references)
        return d


_CHECKS: list[Check] = [
    # ---------------------------------------------------------------- code lint: lookahead
    Check(
        "LA01", "Negative shift (future data) in a feature or signal", "lookahead",
        "`.shift(-n)`, `.pct_change(-n)` or `.diff(-n)` moves values from the future back to today. "
        "That is the normal way to build a *label* (the thing you predict). In a feature or a trading "
        "signal it means the strategy reads tomorrow's data today.",
        "The signal 'knows' part of the return it is about to earn, so hit rate and Sharpe can look "
        "spectacular even when there is no edge at all.",
        "Only use negative shifts to build the label column, and keep that column out of the features. "
        "Features and signals should only use `.shift(k)` with k >= 0.",
        (REF_LEAK, REF_AFML),
    ),
    Check(
        "LA02", "Centered or reversed rolling window", "lookahead",
        "`rolling(..., center=True)` places each window around the current bar, so half of it is in the future. "
        "The same happens when the data is sorted newest-first (`sort_values('date', ascending=False)`, "
        "`sort_index(ascending=False)`, `[::-1]` inside the expression) before `rolling`, `pct_change`, `shift` "
        "or `diff`: each window then covers the bars *after* the current one.",
        "Smoothed values, bands, returns and z-scores reflect prices that had not happened yet, which makes "
        "turning points look predictable.",
        "Drop `center=True` (pandas defaults to a trailing window), and sort ascending by date before any "
        "time-series operation. If you need a newest-first table for display, sort a copy at the end.",
        (REF_LEAK,),
    ),
    Check(
        "LA03", "Backward fill / interpolation", "lookahead",
        "`bfill()`, `fillna(method='bfill')`, `reindex(method='bfill')` and `interpolate()` fill a gap with a "
        "*later* observation.",
        "A missing value is replaced with information from the future (for example, a fundamental that was "
        "only published weeks later), so the strategy trades on data it could not have had.",
        "Use forward fill (`ffill()`) for data that arrives over time, and leave leading gaps as NaN. Drop or "
        "mask rows that have no information yet.",
        (REF_LEAK, REF_AFML),
    ),
    Check(
        "LA04", "Forward-looking as-of merge", "lookahead",
        "`pd.merge_asof(..., direction='forward')` matches each row with the *next* record; "
        "`direction='nearest'` can do the same.",
        "Each row gets the next available record (a later filing, a later quote), which is information from the future.",
        "Use `direction='backward'` (the default) and merge on the timestamp the data became *available*, "
        "for example a filing or acceptance time rather than the fiscal period end.",
        (REF_LEAK,),
    ),
    Check(
        "LA05", "Fitting on the full dataset before a time split", "lookahead",
        "A scaler, PCA, imputer or model is fitted (`fit` / `fit_transform`) on all rows before the "
        "train/test split, or fitted and evaluated on the same rows. A fit on a training subset is not flagged: rows "
        "bounded above by a slice (`.loc[:cut]`, `.iloc[:k]`, `[:k]`, `.loc[start:end]`) or a date mask "
        "(`X[df.date < cut]`, `between`), or a frame named or built that way. A start filter (`.loc['2000':]`, "
        "`X[X.index >= '2000']`) or a test-window mask keeps the later rows, so it is not a training subset. The "
        "check does not verify that the evaluation uses rows after the training subset.",
        "The test period's means, variances and patterns leak into the training step, and an in-sample "
        "fit evaluated on its own data says nothing about the future.",
        "Split by time first. Fit preprocessing and models on the training window only (a sklearn "
        "`Pipeline` inside a walk-forward loop does this for you), then `transform` / `predict` the test window.",
        (REF_AFML, REF_SKLEARN_TS),
    ),
    Check(
        "LA06", "Shuffled or non-temporal cross-validation", "lookahead",
        "`train_test_split` shuffles by default; `KFold(shuffle=True)`, `ShuffleSplit` and default k-fold "
        "CV train on folds that come *after* the fold being tested.",
        "With autocorrelated, overlapping data, neighbours of every test row sit in the training set, so "
        "the model memorizes the test period. Scores look far better than anything achievable live.",
        "Use `train_test_split(..., shuffle=False)` or `TimeSeriesSplit`, ideally with a purge/embargo gap "
        "at least as long as your label horizon.",
        (REF_AFML, REF_SKLEARN_TS),
    ),
    Check(
        "LA07", "Signal and fill on the same bar", "lookahead",
        "The strategy return multiplies a signal computed from bar t's close by bar t's return (close t-1 "
        "to close t); a close-based signal is paired with a return that starts at bar t's open "
        "(`close / open - 1`, `open.shift(-1) / open - 1`, `close - open`); numpy or list slices line the "
        "signal up with the same bar (`sig = c[1:] > c[:-1]` times `c[1:] / c[:-1] - 1`, `zip(c[:-1], c[1:])`); "
        "a bar loop decides on bar i's close (directly, or through a variable set from it earlier in the "
        "iteration) and fills at bar i's open or close, or books bar i's return; an entry or exit priced at bar "
        "t's open is selected by a signal from bar t's close, high, low or volume (`open.where(sig)`, "
        "`open[sig]`, `np.where(sig, open, nan)`, `df.loc[sig, 'entry_px'] = df['open']`); or a "
        "backtest engine fills at the signal bar's close (`cheat_on_close`, `trade_on_close`, vectorbt "
        "`from_signals(..., price=close)` with unshifted entries). Helper functions that multiply their two "
        "arguments are followed to the call site.",
        "The position is credited with the very price move that created the signal. It is a very common leak in quickly written (including AI-generated) strategy code, and it can turn pure noise into a smooth equity curve.",
        "Lag the position: `position = signal.shift(1)` and `strategy = position * returns`. Or equivalently "
        "multiply by the *next* bar's return (`sig[:-1] * ret[1:]` for arrays). Fill at the next bar's open "
        "(`open.shift(-1)`, `open[i + 1]`) if your signal uses the close. In a loop, book the PnL with the "
        "previous bar's position before updating it.",
        (REF_LEAK,),
    ),
    Check(
        "LA08", "Label is the same-bar return", "lookahead",
        "The target (`y`, `target`, `label`) is today's return, while the features are also computed at today's close.",
        "Features and label share the same bar, so the model learns a contemporaneous relationship that "
        "cannot be traded: by the time you know the features, the return has already happened.",
        "Predict the *next* period: `y = returns.shift(-h)` for an h-bar horizon, and drop the last h rows.",
        (REF_AFML,),
    ),
    Check(
        "LA09", "Global normalization over all rows", "lookahead",
        "A z-score or min-max scaling uses the mean, standard deviation, min or max of the *whole* series "
        "(`(x - x.mean()) / x.std()`, `x / x.abs().max()`, `x / x.std()`), a signal compares against a "
        "full-sample mean or quantile, or a percentile rank, `rank()` or `pd.qcut` bins a column over its full "
        "history, including inside functions and helpers that receive the whole column. Per-asset "
        "`groupby(...).transform` of a full-history statistic counts too. Cross-sectional statistics (grouped "
        "by a date or period key such as `date`, `yyyymm`, `jdate`, `fyear`, `month` or `quarter`; `axis=1`; one "
        "date's rows) are not flagged, and inside functions a column must visibly be a time series.",
        "Every row is scaled with statistics that include the future, so thresholds like 'z > 2' are "
        "chosen with hindsight about the series' full range.",
        "Use trailing statistics: `x.rolling(n).mean()` / `.std()` or `.expanding()`, shifted if needed. "
        "For cross-sectional scores, normalize within each date (`groupby(date).transform(...)`).",
        (REF_LEAK, REF_AFML),
    ),
    Check(
        "LA10", "Target leakage (feature built from the label)", "lookahead",
        "A feature or signal is computed from the label or another forward-looking column, or the feature "
        "matrix or a training dataset keeps a forward-looking column (for example, it drops `target` but keeps "
        "`fwd_ret`, or a frame built with `pd.DataFrame`/`pd.concat` stores it). Also flagged: an outcome such as a "
        "forward average, a rolling statistic of forward PnL or a target encoding that is assigned to a signal, "
        "position or weight name, or is transformed (rolling, sign, rank, clip, ratio) and multiplied with a return "
        "(`*`, `.mul`, `np.dot`, `np.where(c, r, -r)`). Forward returns used only as the outcome are not flagged: dropped from the "
        "features before fitting, averaged per date or bucket (`groupby([date, decile])[fwd_ret].mean()`, an IC), or "
        "booked against a position decided earlier (next-open fills). A lead undone by a lag of the same size "
        "(`shift(-H)` then `shift(H)`) is not forward-looking. A value with a neutral name that the lint can't tie "
        "to a model input or a signal is reported as possible.",
        "The model is handed the answer, so its accuracy reflects the leak, not skill. It is one of the "
        "most common causes of 'too good to be true' machine-learning backtests.",
        "Build features only from columns known at the close of t. Select features with an explicit "
        "allow-list instead of 'all columns except the target'.",
        (REF_LEAK,),
    ),
    Check(
        "LA11", "Parameters optimized on the full sample", "overfitting",
        "A loop, nested loop, `itertools.product` grid or dict/list comprehension tries many parameter "
        "values, computes performance on the whole history and keeps the best (max, idxmax, or a ranked table's top row).",
        "The reported result is the maximum of many noisy trials. Even with zero true edge, the best of N "
        "backtests has an expected Sharpe that grows with N, so the reported number is mostly selection bias.",
        "Choose parameters on a training window and report only the untouched test window (walk-forward). "
        "Count every combination you tried and pass it as `n_trials` to `audit_returns` so the Deflated "
        "Sharpe Ratio can correct for it.",
        (REF_PSEUDO, REF_DSR, REF_WHITE),
    ),
    Check(
        "LA12", "Intraday resample stamped at the bar start", "lookahead",
        "`resample('5min')` (and other intraday rules) defaults to `closed='left', label='left'`: the bar "
        "stamped 09:30 contains trades up to 09:34:59. It is reported as medium when those bars are put back "
        "on another time index (`reindex`, `join`, `merge`, `merge_asof`, `concat`, or assigned into a column "
        "of the finer frame), and as possible otherwise. Bars re-stamped before the join (`.shift(1)`, "
        "`index + pd.Timedelta(...)`) or resampled with options passed through `**kwargs` are not flagged.",
        "If you join that bar with anything stamped 09:30, or act on it 'at 09:30', you use up to five "
        "minutes of future data on every bar.",
        "Pass `label='right'` (with `closed='left'`) so each bar is stamped when it is complete, "
        "and make the choice explicit.",
        (REF_PANDAS_RESAMPLE,),
    ),
    Check(
        "LA13", "Future bar used in a loop condition", "lookahead",
        "Inside `for i in ...`, an `if` condition reads `x[i+1]`, `.iloc[i+1]`, a slice starting after i, or a "
        "value read from a later bar earlier in the loop. Offsets set by module constants or parameter defaults "
        "are resolved. Not reported: character loops over a string, and loops whose outputs are only ever used "
        "as a label or target (a triple-barrier labeler). A later bar read only to check for missing data "
        "(`np.isnan`, `pd.isna`, `is None`) is 'possible'. A resting order simulated in the loop is not reported: "
        "an order decided at bar i from past data, filled if bar i+1's low or high reaches its level, with the fill "
        "priced from that level ('possible' if it isn't).",
        "The decision at bar i depends on bar i+1, which is not known yet. Skipping a bar because a later bar is "
        "missing can also drop the trades just before a delisting.",
        "Decide using bars <= i, and use i+1 only to book the fill or the PnL after the decision.",
        (REF_LEAK,),
    ),
    Check(
        "LA14", "Early stopping or model selection on the test set", "overfitting",
        "Early stopping watches the test set (`eval_set=[(X_test, y_test)]` with `early_stopping_rounds`, "
        "`valid_sets=[test]` with an early-stopping callback, Keras `validation_data=(X_test, y_test)` with "
        "`EarlyStopping`; xgboost stops on the last `eval_set` entry), or a loop scores models, parameters or "
        "features on the test set and keeps the best. A set named `holdout` used for early stopping is not flagged "
        "when a separate test set is scored, and walk-forward loops that redefine the test window each iteration "
        "or select by cross-validation are not flagged.",
        "The number of trees or epochs, or the chosen model, is tuned to the test period, so the test score is no "
        "longer out-of-sample. The selection effect grows with the number of rounds or candidates compared.",
        "Split three ways by time: train, validation, test. Early-stop and select on the validation window (or "
        "on folds inside the training period), then evaluate the final model once on the test window. Count "
        "the candidates you compared as `n_trials`.",
        (REF_AFML, REF_PSEUDO, REF_SKLEARN_TS),
    ),
    Check(
        "LA15", "Fundamentals aligned by fiscal period end, not filing date", "lookahead",
        "Quarterly or annual statement data is merged onto prices by its fiscal period-end date (`period_end`, "
        "`fiscalDateEnding`, `datadate`, `calendardate`, `reportperiod`, ...) with `merge_asof`, `merge`, "
        "`reindex(method='ffill')` or `join`, with no reporting lag. A lag counts only when it moves the period-end "
        "column or index itself later (`+ pd.Timedelta(days=90)`, `.add(...)`, `shift(n, freq=...)` with n > 0); a "
        "`merge_asof` tolerance, `MonthEnd(0)` or a time-zone change is not a lag.",
        "Companies file their statements weeks after the quarter ends (US 10-Qs up to 40-45 days, 10-Ks up to "
        "60-90). Aligning on the period end lets the backtest trade on earnings, book values and ratios before they "
        "were published, which is a classic source of inflated value and quality factor returns.",
        "Merge on the date the data became available (filing, acceptance or report date, e.g. Compustat `rdq`, "
        "Sharadar `datekey`), or add a conservative lag to the period end (for example 90 days) before merging, "
        "then use `merge_asof(direction='backward')`.",
        (REF_LEAK, REF_AFML),
    ),
    # ---------------------------------------------------------------- code lint: survivorship, data, costs
    Check(
        "SB01", "Universe built from today's constituents", "survivorship",
        "Tickers come from a *current* list: Wikipedia's S&P 500 table via `pd.read_html`, a "
        "`tickers_sp500()` helper, or a hard-coded list of names; or the panel is filtered to the names whose "
        "last date equals the panel's last date (`groupby('ticker')['date'].max() == df['date'].max()`), which "
        "keeps only the survivors.",
        "Today's members are the companies that survived and grew. Backtesting them over the past "
        "silently excludes everything that went bankrupt, was delisted or dropped out of the index, and "
        "that typically inflates returns.",
        "Use point-in-time index membership (which names were in the index *on each date*), and keep "
        "delisted names with their final returns. Watch out for reused tickers that now point to a different company.",
        (REF_SURV, REF_SURV2),
    ),
    Check(
        "DQ01", "yfinance data caveats", "data",
        "Data is pulled with `yfinance`.",
        "Yahoo has no prices for most delisted tickers (survivorship), restates adjusted history after "
        "every split and dividend, and occasionally has bad prints. The yfinance README describes it as "
        "intended for research and educational use and points to Yahoo's terms of use.",
        "Fine for prototypes. For anything you rely on, use a source with delisted securities and "
        "point-in-time adjustments, and check your universe for missing names.",
        (REF_SURV,),
    ),
    Check(
        "DQ02", "Mixing adjusted and unadjusted prices", "data",
        "Returns are computed from raw `Close` (with `auto_adjust=False`), or adjusted closes are combined "
        "with raw Open/High/Low.",
        "Raw closes show fake crashes on split days and miss dividends. Mixing adjusted and raw prices "
        "creates artificial gaps that a signal can 'predict'.",
        "Compute returns from adjusted prices (or raw prices plus a corporate-actions table), and use the "
        "same adjustment for every price field in a signal.",
        (),
    ),
    Check(
        "CO01", "No transaction costs", "costs",
        "The code computes strategy returns (vectorized, or accumulated in a loop) but never charges commissions, "
        "fees, spread or slippage, or a backtest engine is called without a fee setting. It is always reported as "
        "possible: missing costs make results less realistic but are not a lookahead error, and the returns may "
        "already be net of costs. The message says when a performance figure (Sharpe, CAGR, total return, engine "
        "stats) is computed from the cost-free returns.",
        "High-turnover strategies often have a gross edge smaller than their costs: a daily-rebalanced strategy that trades its whole book every day pays its cost about 250 times a year.",
        "Charge costs on every trade: `net = gross - turnover * cost_bps / 1e4`, with turnover = "
        "sum(|change in weights|). Check the break-even cost with `audit_returns(..., turnover=...)`.",
        (REF_COSTS,),
    ),
    # ---------------------------------------------------------------- returns / trade-log audit
    Check(
        "RA01", "Cost sensitivity and break-even cost", "costs",
        "Net returns at 0/5/10/20 bps per side of turnover, plus the break-even cost: the cost per unit "
        "traded at which the average net return reaches zero.",
        "A backtest that ignores costs, or assumes a cost below what you actually pay, can report a profit "
        "that disappears in real trading.",
        "Make sure the break-even cost is comfortably above your realistic all-in cost (commission plus "
        "half-spread plus impact). Reduce turnover with banding or a slower rebalance.",
        (REF_COSTS,),
    ),
    Check(
        "RA02", "Sharpe ratio significance (Newey-West and Lo-style)", "statistics",
        "The annualized Sharpe ratio; a Newey-West (HAC) t-stat for the mean return, with a bandwidth widened when returns overlap; and a Lo-style (2002) correction: Lo's IID standard error scaled by his autocorrelation correction to annualization, with autocorrelations truncated at the same bandwidth. Significance tests (including the Deflated Sharpe Ratio) use an effective sample size of T divided by the long-run variance ratio.",
        "Autocorrelated returns (from overlapping holding periods, smoothing or stale prices) make the "
        "naive sqrt(252) annualization and the naive t-stat too optimistic.",
        "Read the adjusted t-stats, not the naive one. The verdict needs a Newey-West t of at least 2, and with many trials you need much more (see the Deflated Sharpe Ratio).",
        (REF_LO, REF_NW),
    ),
    Check(
        "RA03", "Deflated Sharpe Ratio (multiple testing)", "overfitting",
        "The probability that the true Sharpe ratio is above the Sharpe you would expect from the *best of "
        "n_trials* pure-noise strategies, adjusting for skewness, fat tails and sample length.",
        "If you tried many variants and report the best one, its Sharpe is inflated by selection alone. "
        "The DSR measures how much of the result survives that correction.",
        "Report n_trials honestly (every parameter set, universe, and rule you tried). The paper's worked "
        "example tests at a 95% confidence level (DSR >= 0.95). Below 0.5, the result can't be told apart from the best of n_trials coin flips.",
        (REF_DSR, REF_HLZ),
    ),
    Check(
        "RA04", "Minimum track record length", "statistics",
        "The number of observations needed before the observed Sharpe is significantly above a reference "
        "Sharpe at 95% confidence, given its skewness and kurtosis.",
        "A short, lucky stretch can produce a high Sharpe. Negative skew and fat tails make the required "
        "track record much longer.",
        "If your sample is shorter than MinTRL, collect more out-of-sample history before trusting the result.",
        (REF_FRONTIER,),
    ),
    Check(
        "RA05", "Sub-period stability", "stability",
        "Performance in each third of the sample and in each calendar year.",
        "An edge that lives in a single period (one regime, one crash, one bubble) is often an "
        "artifact, or may have decayed.",
        "Look for positive results in every third. If the edge is concentrated in one era, ask why "
        "and what would make it persist.",
        (),
    ),
    Check(
        "RA06", "PnL concentration in a few days", "concentration",
        "The share of total PnL that comes from the best 5% of days, compared with what a normal "
        "distribution with the same mean, volatility and length would give.",
        "When a handful of days drive the whole result, the backtest is fragile: missing a few of those "
        "days (or a data error on them) erases it.",
        "Check those days for data errors and one-off events. Make sure the result survives "
        "removing them.",
        (),
    ),
    Check(
        "RA07", "Maximum drawdown and duration", "statistics",
        "The largest peak-to-trough loss of the compounded equity curve and the longest time spent below a previous peak.",
        "Average returns hide the path. A long drawdown is usually where real investors abandon a strategy.",
        "Read the drawdown and its length alongside the return. Live drawdowns are often deeper and longer than backtested ones.",
        (),
    ),
    Check(
        "RA08", "Benchmark-relative performance", "statistics",
        "Excess return over the benchmark you supply, information ratio, beta, and alpha with its Newey-West t-stat. With a benchmark, the verdict needs significant alpha, so leveraged or plain market exposure is not mistaken for skill.",
        "A strategy can look good in absolute terms simply by holding the market in a bull run.",
        "Judge the strategy on its excess return over a fair benchmark (for stocks, at least the "
        "equal-weight universe and SPY).",
        (REF_NW,),
    ),
    Check(
        "RA09", "Implausibly strong results", "lookahead",
        "An annualized Sharpe above 3, or a daily hit rate above 65%, on a strategy evaluated with daily or slower bars.",
        "Results this strong are rare in real trading and are often caused by lookahead (for "
        "example, signal and fill on the same close) or a data error.",
        "Run `check_code` on the strategy and look for same-bar execution, negative shifts and target leakage.",
        (),
    ),
    Check(
        "TL01", "Survivorship hints in a trade log", "survivorship",
        "Heuristics, not proof: every ticker is still trading at the end of a multi-year log, or a ticker is traded before the company that uses the symbol today started trading under it (for example, META before 2022-06-09). Some symbols were used by earlier companies (ARM, SNOW, CEG, META), so read the hint in context.",
        "Both patterns suggest the universe was built from today's names and mapped back in time, which "
        "excludes companies that failed or were acquired.",
        "Rebuild the universe from point-in-time membership, keep delisted names, and map tickers to "
        "the company that used them on each date.",
        (REF_SURV,),
    ),
    Check(
        "TL02", "PnL concentration by ticker", "concentration",
        "The share of total PnL from the top 5 tickers, and the PnL after removing them.",
        "If a few names (often the ones that became today's mega-caps) carry the whole result, the "
        "strategy may just be a hindsight pick.",
        "Make sure the result survives dropping the top contributors.",
        (),
    ),
]

CHECKS: dict[str, Check] = {c.id: c for c in _CHECKS}


def explain(check_id: str | None = None) -> dict:
    """Return the explanation for one check, or the list of all checks if `check_id` is empty."""
    if not check_id:
        return {"checks": [{"id": c.id, "title": c.title, "category": c.category} for c in _CHECKS],
                "disclaimer": DISCLAIMER}
    key = check_id.strip().upper()
    if key not in CHECKS:
        raise KeyError(f"Unknown check id {check_id!r}. Known ids: {', '.join(CHECKS)}")
    d = CHECKS[key].to_dict()
    d["disclaimer"] = DISCLAIMER
    return d


def explain_markdown(check_id: str | None = None) -> str:
    d = explain(check_id)
    if "checks" in d:
        lines = ["# Backtest Auditor checks", "", "| id | title | category |", "|---|---|---|"]
        lines += [f"| {c['id']} | {c['title']} | {c['category']} |" for c in d["checks"]]
        return "\n".join(lines + ["", DISCLAIMER])
    lines = [
        f"# {d['id']}: {d['title']}", "", f"**What it checks.** {d['what']}", "",
        f"**Why it inflates results.** {d['why_it_inflates']}", "", f"**How to fix it.** {d['how_to_fix']}",
    ]
    if d["references"]:
        lines += ["", "**References**"] + [f"- {r}" for r in d["references"]]
    return "\n".join(lines + ["", DISCLAIMER])
