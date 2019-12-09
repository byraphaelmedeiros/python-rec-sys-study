import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

if __name__ == '__main__':
    bank_full = pd.read_csv('evaluating_resources/bank_full_w_dummy_vars.csv')

    print(bank_full.head())
    print(bank_full.info())

    x = bank_full.ix[:, (18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36)].values
    y = bank_full.ix[:, 17].values

    LogReg = LogisticRegression()
    LogReg.fit(x, y)
    y_pred = LogReg.predict(x)

    print(classification_report(y, y_pred))
