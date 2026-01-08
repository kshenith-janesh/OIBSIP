import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (update path)
file_path = (r"C:\Users\Dell\OneDrive\Desktop\OSIB\Project 1\dataset 2\menu.csv")  # Update to your file
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("File not found. Please check the path.")
    exit()

# Inspect data
print("Dataset Head:")
print(df.head())
print("\nActual columns in your dataset:")
print(df.columns.tolist())
print("\nDataset Info:")
print(df.info())
print("\nDataset Description:")
print(df.describe())

# Data Cleaning
df.fillna(df.mean(numeric_only=True), inplace=True)
key_columns = ['Item', 'Calories']  # Updated to match your columns
df.dropna(subset=[col for col in key_columns if col in df.columns], inplace=True)

# No 'Date' column in this dataset, so skip date conversion

df.drop_duplicates(inplace=True)

# Outlier removal
calories_column = 'Calories'
if calories_column in df.columns:
    Q1 = df[calories_column].quantile(0.25)
    Q3 = df[calories_column].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[calories_column] < (Q1 - 1.5 * IQR)) | (df[calories_column] > (Q3 + 1.5 * IQR)))]
else:
    print(f"Warning: Column '{calories_column}' not found. Skipping outlier removal.")

print(f"Cleaned dataset shape: {df.shape}")

# Basic statistics
nutrient_columns = ['Calories', 'Total Fat', 'Protein', 'Carbohydrates']  # Updated to match your exact column names
for col in nutrient_columns:
    if col in df.columns:
        print(f"\n{col} - Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}, Std: {df[col].std():.2f}")
    else:
        print(f"Warning: Column '{col}' not found.")

# Grouped stats
category_column = 'Category'
if category_column in df.columns and calories_column in df.columns:
    grouped_stats = df.groupby(category_column)[calories_column].agg(['mean', 'median', 'std'])
    print(f"\nGrouped Stats by {category_column} (Calories):")
    print(grouped_stats)
else:
    print(f"Warning: Columns '{category_column}' or '{calories_column}' not found. Skipping grouped stats.")

# Time series (skipped since no 'Date' column)
print("No 'Date' column found. Skipping time series.")

# Food analysis
food_column = 'Item'  # Updated to match your column
if food_column in df.columns and calories_column in df.columns:
    food_analysis = df.groupby(food_column)[calories_column].sum().sort_values(ascending=False)
    print(f"\nTop 10 {food_column} by Total Calories:")
    print(food_analysis.head(10))
else:
    print(f"Warning: Columns '{food_column}' or '{calories_column}' not found.")

# Nutrient behavior: Only run if ALL nutrient_columns exist
if category_column in df.columns and all(col in df.columns for col in nutrient_columns):
    nutrient_behavior = df.groupby(category_column)[nutrient_columns].mean()
    print(f"\nAverage Nutrients by {category_column}:")
    print(nutrient_behavior)
else:
    print(f"Warning: '{category_column}' or some nutrient columns not found. Skipping nutrient behavior analysis.")

# Visualizations
sns.set_style('whitegrid')

# Bar chart
if food_column in df.columns and calories_column in df.columns:
    plt.figure(figsize=(10, 6))
    food_analysis.head(10).plot(kind='bar')
    plt.title('Top 10 Items by Calories')
    plt.ylabel('Total Calories')
    plt.xticks(rotation=45)
    plt.show()
else:
    print("Skipping bar chart due to missing columns.")

# Heatmap
numeric_df = df.select_dtypes(include=[np.number])
if not numeric_df.empty:
    corr = numeric_df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Nutrient Correlation Heatmap')
    plt.show()
else:
    print("No numeric columns for heatmap.")

# Scatter plot
fat_column = 'Total Fat'  # Updated to match your column
if calories_column in df.columns and fat_column in df.columns:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=fat_column, y=calories_column)
    plt.title('Calories vs. Total Fat')
    plt.show()
else:
    print(f"Warning: Columns '{calories_column}' or '{fat_column}' not found. Skipping scatter plot.")