# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from textblob import TextBlob  # For sentiment analysis (though we'll use existing sentiment data if available)
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')

# Load the datasets
# Assuming the datasets are in the same directory or provide full paths
apps_df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\OSIB\Project 4-(L2)\datasets\apps.csv")  # Dataset 1: Apps data
reviews_df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\OSIB\Project 4-(L2)\datasets\user_reviews.csv")  # Dataset 2: User reviews data

# Display basic info
print("Apps Dataset Info:")
print(apps_df.info())
print("\nReviews Dataset Info:")
print(reviews_df.info())

# Data Preparation: Clean and correct data types
# For apps_df
# Convert 'Installs' to numeric (remove '+' and ',')
apps_df['Installs'] = apps_df['Installs'].str.replace('[+,]', '', regex=True).astype(float)

# Convert 'Price' to numeric (remove '$')
apps_df['Price'] = apps_df['Price'].str.replace('$', '').astype(float)

# Convert 'Size' to numeric (handle 'Varies with device' and units like 'M', 'k')
# Updated function to handle if Size is already numeric (float) or string
def convert_size(size):
    if pd.isna(size):
        return np.nan
    if isinstance(size, str):
        if size == 'Varies with device':
            return np.nan
        if 'M' in size:
            return float(size.replace('M', '')) * 1e6
        elif 'k' in size:
            return float(size.replace('k', '')) * 1e3
        else:
            try:
                return float(size)
            except ValueError:
                return np.nan
    else:
        # If already numeric (e.g., float), return as is
        return float(size)

apps_df['Size'] = apps_df['Size'].apply(convert_size)

# Convert 'Last Updated' to datetime
apps_df['Last Updated'] = pd.to_datetime(apps_df['Last Updated'], errors='coerce')

# Handle missing values (e.g., drop or fill)
apps_df.dropna(subset=['Rating', 'Installs'], inplace=True)  # Essential columns

# For reviews_df
# The dataset already has 'Sentiment_Polarity', so we'll use that directly
# Drop rows with missing reviews or sentiment
reviews_df.dropna(subset=['Translated_Review', 'Sentiment_Polarity'], inplace=True)

# Category Exploration: Investigate app distribution across categories
category_counts = apps_df['Category'].value_counts()
print("\nApp Distribution by Category:")
print(category_counts)

# Visualization: Bar chart for categories
fig1 = px.bar(category_counts, x=category_counts.index, y=category_counts.values,
              title='App Distribution Across Categories',
              labels={'x': 'Category', 'y': 'Number of Apps'},
              color=category_counts.values, color_continuous_scale='Blues')
fig1.show()

# Metrics Analysis: Examine app ratings, size, popularity (installs), and pricing trends
# Summary statistics
print("\nMetrics Summary:")
print(apps_df[['Rating', 'Size', 'Installs', 'Price']].describe())

# Distribution of Ratings
fig2 = px.histogram(apps_df, x='Rating', nbins=20, title='Distribution of App Ratings',
                    marginal='box', color_discrete_sequence=['skyblue'])
fig2.show()

# Size vs Rating (scatter plot)
fig3 = px.scatter(apps_df, x='Size', y='Rating', color='Category',
                  title='App Size vs Rating', size='Installs', hover_name='App')
fig3.show()

# Popularity (Installs) vs Price
fig4 = px.scatter(apps_df, x='Price', y='Installs', color='Rating',
                  title='Price vs Installs (Popularity)', log_y=True, hover_name='App')
fig4.show()

# Pricing trends by category
price_by_category = apps_df.groupby('Category')['Price'].mean().sort_values(ascending=False)
fig5 = px.bar(price_by_category, x=price_by_category.index, y=price_by_category.values,
              title='Average Price by Category', color=price_by_category.values,
              color_continuous_scale='Reds')
fig5.show()

# Sentiment Analysis: Assess user sentiments through reviews (using existing 'Sentiment_Polarity')
# Merge with apps_df to analyze sentiment by app/category
merged_df = reviews_df.merge(apps_df, on='App', how='left')

# Average sentiment by category (using 'Sentiment_Polarity')
sentiment_by_category = merged_df.groupby('Category')['Sentiment_Polarity'].mean().sort_values()
print("\nAverage Sentiment by Category:")
print(sentiment_by_category)

# Visualization: Sentiment distribution (using 'Sentiment_Polarity')
fig6 = px.histogram(reviews_df, x='Sentiment_Polarity', nbins=30, title='Distribution of Review Sentiments',
                    marginal='rug', color_discrete_sequence=['green'])
fig6.show()

# Sentiment vs Rating (if available)
if 'Sentiment_Polarity' in merged_df.columns and 'Rating' in merged_df.columns:
    fig7 = px.scatter(merged_df, x='Sentiment_Polarity', y='Rating', color='Category',
                      title='Sentiment vs App Rating', hover_name='App')
    fig7.show()

# Interactive Visualization: Combine insights into a dashboard-like view
# Create subplots for key metrics
fig8 = make_subplots(rows=2, cols=2, subplot_titles=('App Categories', 'Rating Distribution',
                                                     'Installs vs Price', 'Sentiment by Category'))

# Add traces
fig8.add_trace(go.Bar(x=category_counts.index, y=category_counts.values, name='Categories'), row=1, col=1)
fig8.add_trace(go.Histogram(x=apps_df['Rating'], nbinsx=20, name='Ratings'), row=1, col=2)
fig8.add_trace(go.Scatter(x=apps_df['Price'], y=apps_df['Installs'], mode='markers', name='Installs vs Price'), row=2, col=1)
fig8.add_trace(go.Bar(x=sentiment_by_category.index, y=sentiment_by_category.values, name='Sentiment'), row=2, col=2)

fig8.update_layout(title_text='Interactive Dashboard: Google Play Store Insights', height=800)
fig8.show()

# Skill Enhancement: Integrate insights from "Understanding Data Visualization"
# - Use color scales effectively (e.g., for emphasis on high values)
# - Ensure interactivity for user exploration (e.g., hover details)
# - Combine multiple views for storytelling (e.g., subplots)
# - Handle data density (e.g., log scales for installs)
# - Provide clear labels and titles for interpretability

# Additional: Correlation heatmap for metrics
corr_matrix = apps_df[['Rating', 'Size', 'Installs', 'Price']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of App Metrics')
plt.show()

# Export cleaned data if needed
apps_df.to_csv('cleaned_apps.csv', index=False)
reviews_df.to_csv('cleaned_reviews.csv', index=False)