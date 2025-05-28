import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/ekstraklasa_2015_2025.csv", sep=",")

X_raw = df.drop(columns=['Result'])
X = pd.get_dummies(X_raw, columns=["Home", "Away"])
y = df['Result']

X['Date'] = pd.to_datetime(X['Date'])
X['Weekday'] = X['Date'].dt.weekday
X['Month'] = X['Date'].dt.month
X = X.drop(columns=['Date', 'Time'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
