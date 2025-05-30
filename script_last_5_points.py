import pandas as pd

df = pd.read_csv("data/ekstraklasa_2015_2025.csv", sep=",")

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

df = df.sort_values('Date').reset_index(drop=True)

def get_last5_points(team, date, current_index):
    recent_matches = df.loc[
        ( ((df['Home'] == team) | (df['Away'] == team)) &
          (df['Date'] < date)
        )
    ].iloc[-5:]

    points = 0
    for _, row in recent_matches.iterrows():
        if row['Home'] == team:
            if row['Result'] == 'H':
                points += 3
            elif row['Result'] == 'D':
                points += 1
        elif row['Away'] == team:
            if row['Result'] == 'A':
                points += 3
            elif row['Result'] == 'D':
                points += 1
    return points

home_points = []
away_points = []

for i, row in df.iterrows():
    home_points.append(get_last5_points(row['Home'], row['Date'], i))
    away_points.append(get_last5_points(row['Away'], row['Date'], i))

df['Home_last5_points'] = home_points
df['Away_last5_points'] = away_points

df.to_csv("data/ekstraklasa_teams_points.csv", sep=",", index=False)

print("Saved as ekstraklasa_teams_points.csv")
