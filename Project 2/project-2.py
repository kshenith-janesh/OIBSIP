# Customer Segmentation Analysis Project (Updated for Your Dataset)
# This script performs customer segmentation using K-means clustering on your e-commerce dataset.
# Assumes your dataset has columns like Income, Age, Recency, MntTotal, etc. (based on your output).
# Adjust the file path as needed.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Step 1: Data Collection and Loading
# Replace with your actual file path.
df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\OSIB\Project 2\ifood_df.csv")  # Update this to your file path, e.g., 'C:/path/to/your/file.csv'

print("Dataset loaded successfully.")
print(df.head())
print(df.info())
print(df.describe())

# Step 2: Data Exploration and Cleaning
# Check for missing values (from your output, there are none)
print("Missing values per column:")
print(df.isnull().sum())

# Handle missing values if any (though your data is clean)
df.fillna(df.median(numeric_only=True), inplace=True)
df.dropna(inplace=True)

# No encoding needed: Your dataset already has numerics and dummies (e.g., marital_*, education_*).
# If you have any remaining categoricals, add pd.get_dummies here.

print(f"Cleaned dataset shape: {df.shape}")
print(df.head())

# Step 3: Descriptive Statistics
# Key metrics (adapted to your columns)
avg_income = df['Income'].mean()
avg_age = df['Age'].mean()
avg_recency = df['Recency'].mean()
avg_mnt_total = df['MntTotal'].mean()
total_customers = df.shape[0]

print(f"\nDescriptive Statistics:")
print(f"Average Income: {avg_income:.2f}")
print(f"Average Age: {avg_age:.2f}")
print(f"Average Recency (days since last purchase): {avg_recency:.2f}")
print(f"Average Total Spending (MntTotal): {avg_mnt_total:.2f}")
print(f"Total Customers: {total_customers}")

# Additional: Frequency of purchases (using NumStorePurchases as proxy)
avg_store_purchases = df['NumStorePurchases'].mean()
print(f"Average Number of Store Purchases: {avg_store_purchases:.2f}")

# Step 4: Customer Segmentation (Using K-means Clustering)
# Select relevant features for clustering (focus on behavioral/demographic data)
# Based on your dataset: Income, Age, Recency, MntTotal, NumWebVisitsMonth, etc.
# Avoid constants like Z_CostContact or irrelevant IDs.
features = ['Income', 'Age', 'Recency', 'MntTotal', 'NumWebVisitsMonth', 'NumStorePurchases']
X = df[features]

# Scale the data for better clustering performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using Elbow Method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Choose K (e.g., 4 based on elbow; adjust based on plot or domain knowledge)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Evaluate clustering quality
sil_score = silhouette_score(X_scaled, df['Cluster'])
print(f"\nSilhouette Score for K={optimal_k}: {sil_score:.2f} (closer to 1 is better)")

# Analyze cluster characteristics
cluster_summary = df.groupby('Cluster')[features].mean()
print(f"\nCluster Summary (Averages):")
print(cluster_summary)

# Step 5: Visualization
# Scatter plot: Income vs. MntTotal (Total Spending), colored by cluster
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Income', y='MntTotal', hue='Cluster', palette='viridis', s=100)
plt.title('Customer Segments: Income vs. Total Spending')
plt.xlabel('Income')
plt.ylabel('Total Spending (MntTotal)')
plt.legend(title='Cluster')
plt.show()

# Bar chart: Average Total Spending by Cluster
plt.figure(figsize=(8, 5))
cluster_avg_spending = df.groupby('Cluster')['MntTotal'].mean()
cluster_avg_spending.plot(kind='bar', color='skyblue')
plt.title('Average Total Spending by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average MntTotal')
plt.xticks(rotation=0)
plt.show()

# Box plot: Age Distribution by Cluster
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Cluster', y='Age', palette='Set2')
plt.title('Age Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Age')
plt.show()

# Additional: Pairplot for selected features (if dataset is small; comment out if too slow)
# sns.pairplot(df, vars=['Income', 'Age', 'MntTotal'], hue='Cluster', palette='viridis')
# plt.suptitle('Pairplot of Key Features by Cluster', y=1.02)
# plt.show()

# Step 6: Insights and Recommendations
print("\nInsights and Recommendations:")
for cluster in sorted(df['Cluster'].unique()):
    subset = df[df['Cluster'] == cluster]
    size = len(subset)
    avg_income = subset['Income'].mean()
    avg_age = subset['Age'].mean()
    avg_recency = subset['Recency'].mean()
    avg_spending = subset['MntTotal'].mean()
    avg_web_visits = subset['NumWebVisitsMonth'].mean()
    
    print(f"\nCluster {cluster} (Size: {size} customers, {size/total_customers*100:.1f}% of total):")
    print(f"  - Avg Income: {avg_income:.1f}")
    print(f"  - Avg Age: {avg_age:.1f}")
    print(f"  - Avg Recency: {avg_recency:.1f} days")
    print(f"  - Avg Total Spending: {avg_spending:.1f}")
    print(f"  - Avg Web Visits/Month: {avg_web_visits:.1f}")
    
    # Example insights (customize based on your data and business context)
    if avg_spending > 1000 and avg_income > 60000:
        print("  - Insight: High-income, high-spending loyal customers. Recommendation: Offer premium products, VIP programs, or personalized upselling to maximize revenue.")
    elif avg_recency > 70:
        print("  - Insight: Low recency (recent purchasers). Recommendation: Re-engagement campaigns like discounts or reminders to bring them back.")
    elif avg_age < 35 and avg_web_visits > 5:
        print("  - Insight: Young, web-active customers. Recommendation: Digital marketing (e.g., social media ads, email newsletters) to boost online purchases.")
    elif avg_spending < 200:
        print("  - Insight: Low-spending segment. Recommendation: Introductory offers, bundles, or targeted promotions to increase purchase value.")
    else:
        print("  - Insight: Balanced segment. Recommendation: Maintain engagement with loyalty rewards and feedback to prevent churn.")

print("\nOverall Recommendations:")
print("- Use these segments for targeted marketing (e.g., email campaigns per cluster).")
print("- Monitor segment changes over time by re-running clustering on new data.")
print("- If silhouette score is low (<0.5), try different K (e.g., 3 or 5) or algorithms like DBSCAN for non-spherical clusters.")
print("- Consider adding more features if available (e.g., product categories) for finer segmentation.")

# End of script
# Run this in a Jupyter notebook or Python environment. Adjust features, K, and insights as needed.