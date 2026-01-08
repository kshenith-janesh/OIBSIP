import pandas as pd
import numpy as np

# Load the dataset (replace with your actual file path)
df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\OSIB\Project 1\dataset 1\retail_sales_dataset.csv")  # Example: Use a CSV like Walmart sales data

# Inspect the data
print(df.head())
print(df.info())
print(df.describe())

# Data Cleaning
# Handle missing values (e.g., fill with mean for numerical, mode for categorical)
df.fillna(df.mean(numeric_only=True), inplace=True)
df.dropna(subset=['Date', 'Total Amount'], inplace=True)  # Drop rows with critical missing data

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Remove duplicates
df.drop_duplicates(inplace=True)

# Check for outliers (e.g., using IQR for Total Amount)
sales_column = 'Total Amount'  # Correct column name
if sales_column in df.columns:
    Q1 = df[sales_column].quantile(0.25)
    Q3 = df[sales_column].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[sales_column] < (Q1 - 1.5 * IQR)) | (df[sales_column] > (Q3 + 1.5 * IQR)))]
else:
    print(f"Warning: Column '{sales_column}' not found. Skipping outlier removal.")

print(f"Cleaned dataset shape: {df.shape}")


# Basic statistics (using the correct column name: 'Total Amount')
sales_column = 'Total Amount'
if sales_column in df.columns:
    mean_sales = df[sales_column].mean()
    median_sales = df[sales_column].median()
    mode_sales = df[sales_column].mode()[0] if not df[sales_column].mode().empty else None  # Handle if no mode exists
    std_sales = df[sales_column].std()

    print(f"Mean Sales: {mean_sales}")
    print(f"Median Sales: {median_sales}")
    print(f"Mode Sales: {mode_sales}")
    print(f"Standard Deviation: {std_sales}")
else:
    print(f"Error: Column '{sales_column}' not found.")

# Grouped stats (e.g., by product category) - using the correct column name: 'Product Category'
category_column = 'Product Category'
if category_column in df.columns and sales_column in df.columns:
    grouped_stats = df.groupby(category_column)[sales_column].agg(['mean', 'median', 'std'])
    print("\nGrouped Stats by Product Category:")
    print(grouped_stats)
else:
    print(f"Error: Columns '{category_column}' or '{sales_column}' not found.")

import matplotlib.pyplot as plt

# Assuming df is already loaded and cleaned, with 'Date' as datetime
# If not, add the loading/cleaning code here

# Set Date as index for time series (ensure 'Date' is datetime)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])  # Convert if not already done
    df.set_index('Date', inplace=True)
else:
    print("Error: 'Date' column not found.")
    exit()

# Resample to monthly sales (using the correct column name: 'Total Amount')
sales_column = 'Total Amount'
if sales_column in df.columns:
    monthly_sales = df[sales_column].resample('M').sum()
    
    # Plot time series
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales, label='Monthly Sales')
    plt.title('Sales Trends Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.legend()
    plt.show()
    
    # Seasonal decomposition skipped (install statsmodels to enable)
    print("Seasonal decomposition skipped due to missing statsmodels. Install with: pip install statsmodels")
else:
    print(f"Error: Column '{sales_column}' not found.")

# Assuming df is already loaded and cleaned
# If not, add the loading/cleaning code here

# Customer analysis: Group by customer ID or demographics (using correct column names)
customer_id_column = 'Customer ID'
sales_column = 'Total Amount'
if customer_id_column in df.columns and sales_column in df.columns:
    customer_analysis = df.groupby(customer_id_column).agg({
        sales_column: 'sum',
        'Quantity': 'sum',
        'Age': 'mean'  # If available
    }).sort_values(sales_column, ascending=False)
    
    print("Top 10 Customers by Total Sales:")
    print(customer_analysis.head(10))  # Top customers
else:
    print(f"Error: Columns '{customer_id_column}' or '{sales_column}' not found.")

# Product analysis: Top-selling products (assuming 'Product Category' for categories)
product_column = 'Product Category'  # Adjust if you meant a different column (e.g., if there's a 'Product Name')
if product_column in df.columns and sales_column in df.columns:
    product_analysis = df.groupby(product_column)[sales_column].sum().sort_values(ascending=False)
    print("\nTop 10 Product Categories by Total Sales:")
    print(product_analysis.head(10))
else:
    print(f"Error: Columns '{product_column}' or '{sales_column}' not found.")

# Purchasing behavior: Average sales by age group
if 'Age' in df.columns and sales_column in df.columns:
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=['<25', '25-35', '35-45', '45-55', '55+'])
    age_behavior = df.groupby('Age_Group')[sales_column].mean()
    print("\nAverage Sales by Age Group:")
    print(age_behavior)
else:
    print(f"Error: Columns 'Age' or '{sales_column}' not found.")

import seaborn as sns

# Bar chart: Top products
plt.figure(figsize=(10, 6))
product_analysis.head(10).plot(kind='bar')
plt.title('Top 10 Products by Sales')
plt.ylabel('Total Sales')
plt.show()

# Line plot: Sales by month (already done in time series)

# Heatmap: Correlation between variables
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Scatter plot: Sales vs. Quantity
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Quantity', y='Sales')
plt.title('Sales vs. Quantity')
plt.show()