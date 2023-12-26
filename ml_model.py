import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Read the CSV file with personality dataset
df = pd.read_csv("C:\Users\imjat\Downloads\codeclause\PA_by_CV\training_dataset.csv")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(), inplace=True)

# Map categorical features to numerical values
gender_mapping = {'Female': 0, 'Male': 1}
df['Gender'] = df['Gender'].map(gender_mapping)

personality_mapping = {'Dependable': 0, 'Serious': 1, 'Responsible': 2, 'Extraverted': 3, 'Lively': 4}
df['Personality'] = df['Personality'].map(personality_mapping)

# Features and target
target = df['Personality'].values
features = df.drop(['Personality'], axis=1).values

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# RandomForestRegressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=0)
rf_model = rf_regressor.fit(x_train, y_train)
rf_predictions = rf_model.predict(x_test)
rf_accuracy = r2_score(y_test, rf_predictions)

# XGBRegressor
xgb_regressor = XGBRegressor(n_estimators=100, random_state=0)
xgb_model = xgb_regressor.fit(x_train, y_train)
xgb_predictions = xgb_model.predict(x_test)
xgb_accuracy = r2_score(y_test, xgb_predictions)

# Choose the better-performing algorithm
better_algorithm = "RandomForest" if rf_accuracy >= xgb_accuracy else "XGBoost"
final_model = rf_model if better_algorithm == "RandomForest" else xgb_model

# Save the final model
import joblib
joblib.dump(final_model, 'personality_model.joblib')
