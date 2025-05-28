import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/ekstraklasa_2015_2025.csv", sep=",")

X = df.drop(columns=['Season', 'Result'])
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
