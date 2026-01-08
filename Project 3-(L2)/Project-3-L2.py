import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import joblib

# Load the dataset
dataset_url = r"C:\Users\Dell\OneDrive\Desktop\OSIB\Project 3-(L2)\creditcard.csv"
df = pd.read_csv(dataset_url)

# Check the column names
print("Columns in the DataFrame:")
print(df.columns)

# 1. **Data Preprocessing:**
# Check for missing values
print("Missing Values Per Column:")
print(df.isnull().sum())

# Handle missing values (fill with median for numerical columns, mode for categorical)
df.fillna(df.median(), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Create a new column 'high_transaction' based on the 'Amount' column
df['high_transaction'] = np.where(df['Amount'] > df['Amount'].median(), 1, 0)  # Flagging high-value transactions

# Check the new dataframe with the 'high_transaction' column
print(df.head())

# 2. **Anomaly Detection with Isolation Forest:**
# We use Isolation Forest to detect anomalies (potential frauds) without labeled data
X = df.drop(columns=['Class'])  # All features except the target column
y = df['Class']  # Target column indicating fraud (1 for fraud, 0 for legit)

iso_forest = IsolationForest(contamination=0.05, random_state=42)  # 5% of data is fraud
y_pred_anomaly = iso_forest.fit_predict(X)

# Anomalies are marked as -1 (fraud) and 1 (legitimate)
df['anomaly'] = np.where(y_pred_anomaly == -1, 1, 0)

# Visualize the anomalies detected by Isolation Forest
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Amount', y='Time', hue='anomaly', data=df, palette='coolwarm')
plt.title('Anomaly Detection (Fraudulent Transactions)')
plt.show()

# 3. **Machine Learning Model - Logistic Regression:**
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)

# Predictions
y_pred_log_reg = log_reg.predict(X_test_scaled)

# Evaluate Logistic Regression
print("\nLogistic Regression Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log_reg):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log_reg))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log_reg))

# 4. **Machine Learning Model - Decision Tree Classifier:**
# Decision Tree model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Predictions
y_pred_tree = decision_tree.predict(X_test)

# Evaluate Decision Tree
print("\nDecision Tree Classifier Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_tree):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_tree))

# 5. **Machine Learning Model - Random Forest Classifier:**
# Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Predictions
y_pred_rf = rf.predict(X_test)

# Evaluate Random Forest
print("\nRandom Forest Classifier Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# 6. **Ensemble Model - Voting Classifier:**
# Combine Logistic Regression, Decision Tree, and Random Forest in a voting classifier
voting_clf = VotingClassifier(estimators=[
    ('log_reg', log_reg),
    ('decision_tree', decision_tree),
    ('rf', rf)
], voting='hard')

# Train and predict with the voting classifier
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)

# Evaluate Ensemble Model
print("\nVoting Classifier Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_voting):.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_voting))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_voting))

# 7. **Feature Importance - Random Forest:**
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.title("Random Forest Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
plt.show()

# Save the trained model (Optional)
joblib.dump(rf, 'fraud_detection_rf_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Final Cleaned Data Sample
print("\nFinal Cleaned Data Sample:")
print(df.head())
