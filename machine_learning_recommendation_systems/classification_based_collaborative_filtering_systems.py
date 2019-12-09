import pandas as pd

from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    bank_full = pd.read_csv('classification_resources/bank_full_w_dummy_vars.csv')

    print(bank_full.head())
    print(bank_full.info())

    x = bank_full.ix[:, (18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36)].values
    y = bank_full.ix[:, 17].values

    LogReg = LogisticRegression()
    LogReg.fit(x, y)

    print(LogReg)

    new_user = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
    y_pred = LogReg.predict(new_user)
    print(y_pred)
