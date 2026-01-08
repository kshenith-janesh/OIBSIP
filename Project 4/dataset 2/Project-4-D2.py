# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download NLTK resources if not already done (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')

# Load the datasets
# Assuming the datasets are in the same directory or provide full paths
apps_df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\OSIB\Project 4\dataset 2\apps.csv")  # Dataset 1: Apps data (not directly used for sentiment, but can be merged if needed)
reviews_df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\OSIB\Project 4\dataset 2\user_reviews.csv")  # Dataset 2: User reviews data (primary for sentiment analysis)

# Display basic info
print("Reviews Dataset Info:")
print(reviews_df.info())
print("\nSample Reviews Data:")
print(reviews_df.head())

# Data Preparation: Clean and prepare text data for sentiment analysis
# Drop rows with missing reviews or sentiment labels
reviews_df.dropna(subset=['Translated_Review', 'Sentiment'], inplace=True)

# Text Preprocessing Function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Apply preprocessing to the review text
reviews_df['Cleaned_Review'] = reviews_df['Translated_Review'].apply(preprocess_text)

# Feature Engineering: Convert text to numerical features using TF-IDF
X = reviews_df['Cleaned_Review']  # Features (text)
y = reviews_df['Sentiment']       # Labels (Positive, Negative, Neutral)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build Pipelines for Machine Learning Models
# 1. Naive Bayes Pipeline
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('classifier', MultinomialNB())
])

# 2. Support Vector Machine (SVM) Pipeline
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('classifier', SVC(kernel='linear', random_state=42))
])

# Train and Evaluate Models
models = {'Naive Bayes': nb_pipeline, 'SVM': svm_pipeline}
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {'Accuracy': accuracy, 'Report': report, 'Confusion Matrix': cm}
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Classification Report:\n{report}")

# Data Visualization: Visualize sentiment analysis results
# 1. Sentiment Distribution (Original Labels)
sentiment_counts = reviews_df['Sentiment'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Distribution of Sentiment Labels')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# 2. Confusion Matrix Heatmap for Best Model (e.g., SVM)
best_model = 'SVM'  # Assuming SVM performs better; adjust based on results
cm = results[best_model]['Confusion Matrix']
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title(f'Confusion Matrix for {best_model}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 3. Accuracy Comparison Bar Chart
accuracies = [results[model]['Accuracy'] for model in results]
model_names = list(results.keys())
plt.figure(figsize=(6, 4))
sns.barplot(x=model_names, y=accuracies, palette='coolwarm')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# Optional: Predict Sentiment on New Text (Example)
def predict_sentiment(text, model=svm_pipeline):
    cleaned_text = preprocess_text(text)
    prediction = model.predict([cleaned_text])
    return prediction[0]

# Example Prediction
sample_text = "This app is amazing and works perfectly!"
print(f"\nSample Prediction for '{sample_text}': {predict_sentiment(sample_text)}")

# Export cleaned data or model if needed
reviews_df.to_csv('cleaned_reviews_with_sentiment.csv', index=False)

# Skill Enhancement: Integrate insights from NLP and ML best practices
# - Preprocessing improves model input quality.
# - TF-IDF captures important words/phrases.
# - Evaluation metrics ensure model reliability.
# - Visualizations aid in interpreting results and model performance.