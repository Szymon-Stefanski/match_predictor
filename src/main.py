import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("../data/ekstraklasa_2015_2025.csv", sep=",")

X_raw = df.drop(columns=['Result'])
X = pd.get_dummies(X_raw, columns=["Home", "Away", "Season"])
y = df['Result']

X['Date'] = pd.to_datetime(X['Date'], dayfirst=True)
X['Weekday'] = X['Date'].dt.weekday
X['Month'] = X['Date'].dt.month
X = X.drop(columns=['Date', 'Time'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

clf = LogisticRegression(max_iter=10000, random_state=0)
clf.fit(X_train, y_train)

acc = accuracy_score(y_test, clf.predict(X_test)) * 100
print(f"Logistic Regression model accuracy: {acc:.2f}%")
