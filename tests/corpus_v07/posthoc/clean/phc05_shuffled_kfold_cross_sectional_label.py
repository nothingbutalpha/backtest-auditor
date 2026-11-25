# expect: none
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

loans = pd.read_parquet("loan_book_snapshot.parquet")
X = loans[["ltv", "dti", "fico", "income"]]
y = loans["defaulted"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
print(cross_val_score(LogisticRegression(max_iter=500), X, y, cv=skf).mean())
