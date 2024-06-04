import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load your dataset
df = pd.read_csv('DataSet.csv')
questions = df['question'].values.tolist()
responses = df['response'].values.tolist()

# Create the dataset
dataset = {
    'questions': questions,
    'answers': responses
}

# Preprocess the data
nltk.download('punkt')
nltk.download('stopwords')


def clean(text):
    # Remove leading and trailing whitespaces
    text = text.strip()

    # Remove '|' characters from the beginning and end of the string
    text = text.strip('|')

    return text


def preprocess(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Define and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

    # Apply stemming using Porter Stemmer
    tokens = [PorterStemmer().stem(token) for token in tokens]

    # Join the processed tokens back into a string
    processed_text = ' '.join(tokens)

    return processed_text


cleaned_questions = [clean(question) for question in dataset['questions']]
cleaned_answers = [clean(answer) for answer in dataset['answers']]
dataset['answers'] = cleaned_answers
processed_questions = [preprocess(question) for question in cleaned_questions]

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_questions)


def get_response(user_input):
    user_input = preprocess(user_input)
    user_tfidf = vectorizer.transform([user_input])

    # Calculate cosine similarity between user input and each question
    similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

    # Find the index of the most similar question
    max_similarity_index = similarities.argmax()

    # Return the corresponding answer
    return dataset['answers'][max_similarity_index]


# Flask routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get_response', methods=['POST'])
def get_response_route():
    user_input = request.form['user_input']
    response = get_response(user_input)
    return {'response': response}


# Simple chat loop
if __name__ == '__main__':
    app.run(debug=True)
