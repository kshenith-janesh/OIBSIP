import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Step 1: Load Dataset
# ------------------------------
df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\OSIB\Project 1\dataset 1\Dataset\retail_sales_dataset.csv")

# Inspect the data
print(df.head())
print(df.info())
print(df.describe())

# ------------------------------
# Step 2: Data Cleaning
# ------------------------------
# Remove leading/trailing spaces in column names
df.columns = df.columns.str.strip()

# Fill missing numeric values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Drop rows with critical missing data
df.dropna(subset=['Date', 'Total Amount'], inplace=True)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Remove duplicates
df.drop_duplicates(inplace=True)

# Outlier removal for 'Total Amount' using IQR
sales_column = 'Total Amount'
if sales_column in df.columns:
    Q1 = df[sales_column].quantile(0.25)
    Q3 = df[sales_column].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[sales_column] < (Q1 - 1.5 * IQR)) | (df[sales_column] > (Q3 + 1.5 * IQR)))]
else:
    print(f"Warning: Column '{sales_column}' not found. Skipping outlier removal.")

print(f"Cleaned dataset shape: {df.shape}")

# ------------------------------
# Step 3: Basic Statistics
# ------------------------------
if sales_column in df.columns:
    mean_sales = df[sales_column].mean()
    median_sales = df[sales_column].median()
    mode_sales = df[sales_column].mode()[0] if not df[sales_column].mode().empty else None
    std_sales = df[sales_column].std()

    print(f"Mean Sales: {mean_sales}")
    print(f"Median Sales: {median_sales}")
    print(f"Mode Sales: {mode_sales}")
    print(f"Standard Deviation: {std_sales}")

# Grouped stats by Product Category
category_column = 'Product Category'
if category_column in df.columns and sales_column in df.columns:
    grouped_stats = df.groupby(category_column)[sales_column].agg(['mean', 'median', 'std'])
    print("\nGrouped Stats by Product Category:")
    print(grouped_stats)

# ------------------------------
# Step 4: Time Series Analysis
# ------------------------------
df.set_index('Date', inplace=True)
monthly_sales = df[sales_column].resample('M').sum()

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales, marker='o', label='Monthly Sales')
plt.title('Sales Trends Over Time')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.grid(True)
plt.legend()
plt.show()

print("Seasonal decomposition skipped due to missing statsmodels. Install with: pip install statsmodels")

# ------------------------------
# Step 5: Customer Analysis
# ------------------------------
customer_id_column = 'Customer ID'
if customer_id_column in df.columns and sales_column in df.columns:
    customer_analysis = df.groupby(customer_id_column).agg({
        sales_column: 'sum',
        'Quantity': 'sum',
        'Age': 'mean'
    }).sort_values(sales_column, ascending=False)

    print("Top 10 Customers by Total Sales:")
    print(customer_analysis.head(10))

# ------------------------------
# Step 6: Product Analysis
# ------------------------------
if category_column in df.columns and sales_column in df.columns:
    product_analysis = df.groupby(category_column)[sales_column].sum().sort_values(ascending=False)
    print("\nTop 10 Product Categories by Total Sales:")
    print(product_analysis.head(10))

# ------------------------------
# Step 7: Purchasing Behavior by Age Group
# ------------------------------
if 'Age' in df.columns and sales_column in df.columns:
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100],
                             labels=['<25', '25-35', '35-45', '45-55', '55+'])
    age_behavior = df.groupby('Age_Group')[sales_column].mean()
    print("\nAverage Sales by Age Group:")
    print(age_behavior)

# ------------------------------
# Step 8: Visualization
# ------------------------------
# Bar chart: Top Product Categories
plt.figure(figsize=(10, 6))
product_analysis.head(10).plot(kind='bar')
plt.title('Top 10 Product Categories by Sales')
plt.ylabel('Total Sales')
plt.show()

# Heatmap: Correlation between numeric variables
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Scatter plot: Sales vs Quantity (fixed column name)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Quantity', y='Total Amount')  # Corrected
plt.title('Sales vs Quantity')
plt.xlabel('Quantity')
plt.ylabel('Sales')
plt.show()
