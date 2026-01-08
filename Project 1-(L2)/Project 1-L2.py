import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Loading and Cleaning
df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\OSIB\Project 1-(L2)\Housing.csv")  # Adjust path if needed

print(df.head())
print(df.info())
print(df.describe())

# Data Cleaning
df.fillna(df.median(numeric_only=True), inplace=True)
df.dropna(inplace=True)

# Encode categoricals
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Remove outliers for 'price'
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['price'] < (Q1 - 1.5 * IQR)) | (df['price'] > (Q3 + 1.5 * IQR)))]

print(f"Cleaned dataset shape: {df.shape}")

# 2. Descriptive Statistics
mean_price = df['price'].mean()
median_price = df['price'].median()
std_price = df['price'].std()

print(f"Mean Price: {mean_price}")
print(f"Median Price: {median_price}")
print(f"Std Dev Price: {std_price}")

# Correlations
corr_matrix = df.corr()
print(corr_matrix['price'].sort_values(ascending=False))

# 3. Feature Analysis
# Average price by bedrooms
bedroom_analysis = df.groupby('bedrooms')['price'].mean()
print(bedroom_analysis)

# 4. Visualization
# Histogram: Price distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['price'], bins=30, kde=True)
plt.title('Distribution of Housing Prices')
plt.show()

# Scatter plot: Price vs. Area
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='area', y='price')
plt.title('Price vs. Area')
plt.show()

# Heatmap: Correlations
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Box plot: Price by Furnishing Status (using encoded column)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='furnishingstatus_semi-furnished', y='price')  # Example; adjust if needed
plt.title('Price by Furnishing Status')
plt.show()

# 5. Insights and Recommendations
# Based on correlations, e.g., if 'area' and 'bathrooms' correlate highly with 'price'
print("Top correlated features with price:")
print(corr_matrix['price'].sort_values(ascending=False).head(5))
