'''
This code was created using gemini, an AI programmer
Prompt: what would be the best way to split this csv into usable sections of each game franchise
*EA_games_clean.csv was given prior to the prompt
'''


import pandas as pd
import numpy as np

df = pd.read_csv("CSV_files/EA_sports_only.csv")

# Define the regex conditions for each franchise
conditions = [
    df['name'].str.contains('FIFA|World Cup|UEFA|Football Academy', case=False, na=False),
    df['name'].str.contains('Madden|NFL Head Coach|Arena Football', case=False, na=False),
    df['name'].str.contains('NCAA Football', case=False, na=False),
    df['name'].str.contains('NBA', case=False, na=False),
    df['name'].str.contains('NCAA March Madness|NCAA Basketball', case=False, na=False),
    df['name'].str.contains('NHL', case=False, na=False),
    df['name'].str.contains('PGA Tour|Tiger Woods', case=False, na=False),
    df['name'].str.contains('MVP.*Baseball|Triple Play', case=False, na=False),
    df['name'].str.contains('Fight Night|Knockout Kings|UFC|MMA|FaceBreaker', case=False, na=False),
    df['name'].str.contains('NASCAR|F1 |F1 Career|Andretti|Superbike', case=False, na=False)
]

choices = [
    'FIFA/Soccer', 'Madden/NFL', 'NCAA Football', 'NBA', 'NCAA Basketball', 
    'NHL', 'PGA Tour', 'Baseball', 'Fighting', 'Racing'
]

# Create the new column, defaulting to 'Other' for anything unmatched (e.g., SSX, Rugby)
df['franchise'] = np.select(conditions, choices, default='Other')

# Split the master dataframe into a dictionary of smaller dataframes
franchise_dfs = {franchise: group for franchise, group in df.groupby('franchise')}

# 1. Sort chronologically so historical shifts work correctly
df = df.sort_values(by=['franchise', 'date']).reset_index(drop=True)

# 2. Shift by 1 within each franchise to get the previous game's sales
df['prev_game_sales'] = df.groupby('franchise')['total'].shift(1)

# 3. Get the previous game's critic score
df['prev_critic_score'] = df.groupby('franchise')['critic'].shift(1)

# 4. Calculate a rolling 3-game average of past sales
# Note: We shift first, THEN roll, to ensure we don't include the current game's sales (Data Leakage)
df['rolling_3_avg_sales'] = (
    df.groupby('franchise')['total']
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)

# 5. Calculate the days since the last release in the franchise
df['date'] = pd.to_datetime(df['date'])
df['days_since_last_release'] = df.groupby('franchise')['date'].diff().dt.days

df.to_csv('Test_AI_data.csv', index=False)
