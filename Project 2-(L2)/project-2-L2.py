import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Data Collection and Loading

try:
    df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\OSIB\Project 2-(L2)\WineQT.csv")  # Use default ',' separator
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Dataset not found. Using sklearn fallback (synthetic approximation).")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1599, n_features=11, n_classes=6, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(11)])
    df['quality'] = y

print(df.head())  # Check the first few rows
print(df.info())  # Get info about the dataset
print(df.describe())

# Step 2: Data Exploration and Cleaning
# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Handle missing values (if any)
df.fillna(df.median(numeric_only=True), inplace=True)

# Check target distribution
print("Quality distribution:")
print(df['quality'].value_counts())

# Step 3: Descriptive Statistics
# Key metrics
avg_fixed_acidity = df['fixed acidity'].mean() if 'fixed acidity' in df.columns else df.iloc[:, 0].mean()
avg_density = df['density'].mean() if 'density' in df.columns else df.iloc[:, 7].mean()
avg_alcohol = df['alcohol'].mean() if 'alcohol' in df.columns else df.iloc[:, 10].mean()

print(f"\nDescriptive Statistics:")
print(f"Average Fixed Acidity: {avg_fixed_acidity:.2f}")
print(f"Average Density: {avg_density:.4f}")
print(f"Average Alcohol: {avg_alcohol:.2f}")
print(f"Total Samples: {len(df)}")

# Step 4: Data Visualization
# Histogram: Quality distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['quality'], bins=10, kde=True)
plt.title('Distribution of Wine Quality')
plt.show()

# Scatter plot: Alcohol vs. Quality
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='alcohol', y='quality')
plt.title('Alcohol vs. Quality')
plt.show()

# Heatmap: Correlations
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Box plot: Acidity by Quality
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='quality', y='fixed acidity')
plt.title('Fixed Acidity by Quality')
plt.show()

# Step 5: Model Training and Evaluation
# Prepare data
features = df.drop('quality', axis=1)
target = df['quality']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SGD Classifier': SGDClassifier(random_state=42),
    'SVC': SVC(random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'Accuracy': accuracy,
        'Classification Report': classification_report(y_test, y_pred, zero_division=0),
        'Confusion Matrix': confusion_matrix(y_test, y_pred)
    }
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(results[name]['Classification Report'])

# Step 6: Insights and Recommendations
print("\nModel Comparison:")
for name, res in results.items():
    print(f"{name}: Accuracy = {res['Accuracy']:.2f}")

# Example insights
best_model = max(results, key=lambda x: results[x]['Accuracy'])
print(f"\nBest Model: {best_model} with accuracy {results[best_model]['Accuracy']:.2f}")
print("Insights:")
print("- Random Forest often performs well due to handling non-linear relationships.")
print("- SGD is efficient for large datasets but may need tuning.")
print("- SVC is good for small datasets but can be slow.")
print("Recommendations:")
print("- Use the best model for predictions; tune hyperparameters (e.g., via GridSearchCV) for better accuracy.")
print("- Focus on highly correlated features (e.g., alcohol) for feature engineering.")
print("- Visualize misclassifications using confusion matrices for further improvement.")
