import pandas as pd

data = pd.read_csv('EA_games_clean.csv')
original = pd.read_csv('vgchartz.csv')

'''
Code block was ran once to clean the data and save it to a new CSV file.
This block no longer needs to be run, but is kept here for reference.
'''
# 1. Convert the string column to official Pandas datetime objects
data['date'] = pd.to_datetime(original['date'])

# 2. Extract the components into distinct numeric columns

data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['day_of_year'] = data['date'].dt.dayofyear


# 3. Drop the original 'date' column 
data = data.drop('date', axis=1)


# Convert the categorical column into multiple binary (1/0) columns
data = pd.get_dummies(data, columns=['platform_grouped'], dtype=int, drop_first=True)
data = pd.get_dummies(data, columns=['developer'], dtype=int, drop_first=True)

data.to_csv('EA_games_clean.csv', index=False)

