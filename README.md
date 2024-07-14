# Intel-products-Sentiment-analysis-from-Online-Reviews
Sentiment analysis, also known as opinion mining, is the process of determining the emotional tone behind a series of words. Importance: Helps in understanding the opinions expressed in reviews, tweets, and other user-generated content.

import pandas as pd 
import numpy as np 
import re 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('punkt') 
nltk.download('stopwords') 
nltk.download('wordnet')
Step 3: Load and Preprocess Data
# Load data
data = pd.read_csv('intel_reviews.csv')
# Preprocessing function
def preprocess_text(text): 
# Lowercase
 text = text.lower()
 # Remove punctuation 
 text = re.sub(r'[^\w\s]', '', text)
 # Tokenize 
tokens = word_tokenize(text)
 # Remove stop words
 stop_words = set(stopwords.words('english')) 
tokens = [word for word in tokens if word not in stop_words] 
# Lemmatize 
lemmatizer = WordNetLemmatizer() 
tokens = [lemmatizer.lemmatize(word) for word in tokens] 
return ' '.join(tokens) 
# Apply preprocessing 
data['cleaned_review'] = data['review'].apply(preprocess_text)
Step 4: Feature Extraction
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['cleaned_review'], data['sentiment'], test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
Step 5: Train Naive Bayes Classifier
# Train Naive Bayes classifier 
nb_classifier = MultinomialNB() 
nb_classifier.fit(X_train_tfidf, y_train) 
# Predict on test data 
y_pred = nb_classifier.predict(X_test_tfidf) 
# Evaluate model 
accuracy = accuracy_score(y_test, y_pred) 
conf_matrix = confusion_matrix(y_test, y_pred) 
class_report = classification_report(y_test, y_pred) 
print(f'Accuracy: {accuracy}') 
print('Confusion Matrix:') 
print(conf_matrix) 
print('Classification Report:') 
print(class_report)
Data Collection Method Using Web Scraping (OPTIONAL)
from bs4 import BeautifulSoup 
import requests 
# Example scraping function for a single product page 
def scrape_reviews(url): 
response = requests.get(url) 
soup = BeautifulSoup(response.content, 'html.parser') 
reviews = [] 
for review in soup.find_all('div', class_='review'): 
text = review.find('p').get_text() 
sentiment = 'positive' if 'positive' in review['class'] else 'negative' reviews.append({'review': text, 'sentiment': sentiment}) 
return reviews 
url = 'http://example.com/product-reviews' 
reviews = scrape_reviews(url) 
reviews_df = pd.DataFrame(reviews) 
reviews_df.to_csv('intel_reviews.csv', index=False)
