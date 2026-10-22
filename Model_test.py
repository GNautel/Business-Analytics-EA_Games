import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


sns.set_style("whitegrid")


data = pd.read_csv('EA_games_clean.csv')

'''
Code block was ran once to clean the data and save it to a new CSV file.
This block no longer needs to be run, but is kept here for reference.

# 1. Convert the string column to official Pandas datetime objects
data['date'] = pd.to_datetime(data['date'])

# 2. Extract the components into distinct numeric columns
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

# 3. Drop the original 'date' column 
data = data.drop('date', axis=1)

# 4. Drop the columns that are not needed for modeling. Other sales numbers from different regions will incorrectly increase model accuracy 
data = data.drop(['name', 'platform'
                  , 'publisher', 'america', 'europe', 'japan', 'other'], axis=1)

# Convert the categorical column into multiple binary (1/0) columns
data = pd.get_dummies(data, columns=['platform_grouped'], dtype=int, drop_first=True)
data = pd.get_dummies(data, columns=['developer'], dtype=int, drop_first=True)

data.to_csv('EA_games_clean.csv', index=False)
'''

# Start XGBoost model training and evaluation

X = data.drop('total', axis=1)
y = data['total']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=41
)

model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("First 10 Predictions:", y_pred[:10])

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared: {r2:.2f}")


# Create a scatter plot of Actual vs. Predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, color='blue')

# Plot the "Perfect Prediction" line (y = x)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

plt.title('Actual vs. Predicted Values')
plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values (y_pred)')
plt.show()