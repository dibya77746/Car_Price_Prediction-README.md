import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Load Dataset
df = pd.read_csv("quikr_car.csv")

# Data Cleaning
df = df[df['year'].str.isnumeric()]
df['year'] = df['year'].astype(int)

df = df[df['Price'] != 'Ask For Price']
df['Price'] = df['Price'].str.replace(",", "", regex=True).astype(int)

df = df[df['kms_driven'].str.replace(',', '', regex=True).str.replace(' kms', '', regex=True).str.isnumeric()]
df['kms_driven'] = df['kms_driven'].str.replace(',', '', regex=True).str.replace(' kms', '', regex=True)
df['kms_driven'] = df['kms_driven'].astype(int)

df = df[~df['fuel_type'].isnull()]

# Clean names
df['name'] = df['name'].str.split(" ").str.slice(0, 3).str.join(" ")

# One-hot encode fuel_type
df = pd.get_dummies(df, columns=["fuel_type"], drop_first=True)

# Build feature list dynamically
feature_cols = ['year', 'kms_driven']
for col in ['fuel_type_Diesel', 'fuel_type_Petrol', 'fuel_type_LPG', 'fuel_type_CNG']:
    if col in df.columns:
        feature_cols.append(col)

X = df[feature_cols]
y = df['Price']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction for first 10 rows
first_ten = df.head(10).copy()
first_ten_features = first_ten[feature_cols]
first_ten['Predicted_Price'] = model.predict(first_ten_features).astype(int)

# Show all details with predicted price
print("\nFirst 10 cars with predicted prices:\n")
print(first_ten[['name', 'company', 'year', 'Price', 'kms_driven'] + [col for col in df.columns if col.startswith('fuel_type_')] + ['Predicted_Price']])

# --- Figure 1: Actual vs Predicted Car Prices ---
y_pred = model.predict(X_test)
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Figure 2: Distribution of Actual Car Prices ---
plt.figure(figsize=(6,4))
plt.hist(y, bins=30, color='skyblue', edgecolor='black')
plt.xlabel("Car Price")
plt.ylabel("Frequency")
plt.title("Distribution of Actual Car Prices")
plt.tight_layout()
plt.show()

# --- Figure 3: Year-wise Average Car Price ---
plt.figure(figsize=(8,4))
df.groupby('year')['Price'].mean().plot(kind='bar', color='orange')
plt.xlabel("Year")
plt.ylabel("Average Price")
plt.title("Year-wise Average Car Price")
plt.tight_layout()
plt.show()

# --- Figure 4: Company-wise Average Car Price ---
plt.figure(figsize=(10,4))
df.groupby('company')['Price'].mean().sort_values(ascending=False).head(15).plot(kind='bar', color='green')
plt.xlabel("Company")
plt.ylabel("Average Price")
plt.title("Top 15 Companies by Average Car Price")
plt.tight_layout()
plt.show()