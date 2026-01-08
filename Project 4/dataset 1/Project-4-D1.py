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

# Load the dataset (use a small subset for testing)
twitter_df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\OSIB\Project 4\dataset 1\Twitter_Data.csv")  # Dataset 1: Twitter data

# Display basic info
print("Twitter Dataset Info:")
print(twitter_df.info())
print("\nSample Twitter Data:")
print(twitter_df.head())

# Data Preparation: Clean and prepare text data for sentiment analysis
# Adjusted column names to 'clean_text' for text and 'sentiment' for labels
text_column = 'clean_text'  # Change to your actual text column name
label_column = 'category'  # 'category' column seems to be your label, not 'sentiment'

# Drop rows with missing text or labels (use the correct column names)
twitter_df.dropna(subset=[text_column, label_column], inplace=True)

# Text Preprocessing Function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters, numbers, and punctuation (including Twitter-specific like @, #)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Test preprocessing on a small subset (first 1000 rows for faster feedback)
twitter_df_small = twitter_df.head(1000)  # Use only the first 1000 rows for initial testing
twitter_df_small['Cleaned_Tweet'] = twitter_df_small[text_column].apply(preprocess_text)

# Feature Engineering: Convert text to numerical features using TF-IDF
X = twitter_df_small['Cleaned_Tweet']  # Features (text)
y = twitter_df_small[label_column]     # Labels (e.g., Positive, Negative, Neutral)

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
    print(f"\nTraining {name} on the small dataset...")
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
sentiment_counts = twitter_df[label_column].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Distribution of Sentiment Labels in Twitter Data')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# 2. Confusion Matrix Heatmap for Best Model (e.g., SVM)
best_model = 'SVM'  # Assuming SVM performs better; adjust based on results
cm = results[best_model]['Confusion Matrix']
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
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
sample_tweet = "This product is fantastic and exceeded my expectations! #happy"
print(f"\nSample Prediction for '{sample_tweet}': {predict_sentiment(sample_tweet)}")

# Export cleaned data or model if needed
twitter_df_small.to_csv('cleaned_twitter_with_sentiment.csv', index=False)
