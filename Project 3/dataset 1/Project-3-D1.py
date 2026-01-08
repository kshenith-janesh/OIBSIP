import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (replace with your dataset URL or file path)
dataset_url = r"C:\Users\Dell\OneDrive\Desktop\OSIB\Project 3\dataset 1\AB_NYC_2019.csv"
df = pd.read_csv(dataset_url)

# 1. **Data Integrity:**
# Check if any column contains invalid entries (e.g., wrong types, unexpected values).
print("Data Types of Columns:")
print(df.dtypes)

# Identify non-numeric columns that may require encoding or further cleaning
print("\nNon-Numeric Columns (possible encoding needed):")
print(df.select_dtypes(include=['object']).columns)

# Check for inconsistent categorical values
print("\nUnique Values in Categorical Columns:")
for column in df.select_dtypes(include=['object']).columns:
    print(f"\n{column}:")
    print(df[column].unique())

# 2. **Missing Data Handling:**
# Check for missing values
print("\nMissing Values Per Column:")
print(df.isnull().sum())

# Handle missing values:
# Option 1: Drop rows with missing target values (e.g., 'price')
df = df.dropna(subset=['price'])  # Dropping rows where 'price' is missing

# Option 2: Fill missing numerical data with median
# Only apply this to numerical columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Option 3: Fill categorical columns with the mode (most frequent value)
for column in df.select_dtypes(include=['object']).columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

# Verify missing values handled
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 3. **Duplicate Removal:**
# Identify and remove duplicate rows
print(f"\nNumber of Duplicate Rows: {df.duplicated().sum()}")
df = df.drop_duplicates()

# Verify if duplicates were removed
print(f"\nNumber of Duplicate Rows After Removal: {df.duplicated().sum()}")

# 4. **Standardization:**
# Check if there are any inconsistencies in formatting for numeric columns
# Example: Check for negative values where there shouldn't be any
for col in numeric_columns:
    negative_values = df[df[col] < 0]
    if not negative_values.empty:
        print(f"\nNegative values found in column '{col}':")
        print(negative_values)

# Apply transformations to fix issues (e.g., set negative values to 0 if they don't make sense)
df[numeric_columns] = df[numeric_columns].apply(lambda x: x.clip(lower=0))

# 5. **Outlier Detection:**
# Visualize outliers using boxplots or scatter plots
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numeric_columns])
plt.title("Boxplot for Outlier Detection")
plt.show()

# Detect outliers using the IQR method (Interquartile Range)
Q1 = df[numeric_columns].quantile(0.25)
Q3 = df[numeric_columns].quantile(0.75)
IQR = Q3 - Q1
outliers = ((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR)))
print(f"\nNumber of Outliers Detected: {outliers.sum().sum()}")

# Optionally, remove outliers
df = df[~outliers.any(axis=1)]

# Verify outliers removal
outliers_after = ((df[numeric_columns] < (Q1 - 1.5 * IQR)) | (df[numeric_columns] > (Q3 + 1.5 * IQR)))
print(f"\nNumber of Outliers After Removal: {outliers_after.sum().sum()}")

# Final Cleaned DataFrame
print("\nFinal Cleaned Data Sample:")
print(df.head())

# Save cleaned data if needed
df.to_csv(r"C:\Users\Dell\OneDrive\Desktop\OSIB\Project 3\dataset 1\AB_NYC_2019_cleaned.csv", index=False)
