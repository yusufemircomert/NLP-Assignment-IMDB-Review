import re
import string
import math
import pandas as pd
from collections import Counter
from datasets import load_dataset
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Download and load the stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Used AI to make comments to the code.

# Class to load and preprocess the texts
def preprocess_text(text): # Same function as Naive Bayes's preprocess_text
    # Removing punctuation symbols
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # Removing digits
    text = re.sub(r"\d+", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove stop words
    # Using NLTK stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = " ".join(words)

    text = re.sub(r"\s+", " ", text).strip()

    return text

# Function to load and preprocess the dataset
def prepare_dataset():
    dataset = load_dataset("imdb")
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    # Preprocess the text data
    train_df["text"] = train_df["text"].apply(preprocess_text)
    test_df["text"] = test_df["text"].apply(preprocess_text)

    return train_df, test_df

# Function to calculate bias scores
def bias_scores(train_df): # Used AI in the for loop where word in all_words
    # Extract the positive and negative samples
    pos_texts = train_df[train_df["label"] == 1]["text"]
    neg_texts = train_df[train_df["label"] == 0]["text"]

    # Count the frequencies of each word
    pos_counter = Counter()
    neg_counter = Counter()

    # Count the words in positive and negative texts
    for text in pos_texts:
        pos_counter.update(text.split())
    for text in neg_texts:
        neg_counter.update(text.split())

    # Count the total number of words in positive and negative texts
    all_words = set(pos_counter.keys()).union(set(neg_counter.keys()))
    scores = []

    # Calculate the bias score for each word
    for word in all_words:
        fp = pos_counter[word]
        fn = neg_counter[word]
        ft = fp + fn
        
        score = abs(fp - fn) / ft * math.log(ft)
        scores.append((word, fp, fn, ft, score))

    # Sort the scores based on the bias score
    scores.sort(key=lambda x: x[4], reverse=True)
    return scores[:10000]

# Function to vectorize the text data using the top 10000 words
# and returns the train and test sets
def vectorize_text(scores, train_df, test_df):
    top_words = [t[0] for t in scores]
    vectorizer = CountVectorizer(vocabulary=top_words)

    X_train = vectorizer.transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    return X_train, X_test, y_train, y_test

# Function to train the Logistic Regression model and plot the results
def train(X_train, X_test, y_train, y_test):
    train_accuracies = []
    test_accuracies = []

    # Train the model with varying max_iter values   
    for i in range(1, 26):
        lr_model = LogisticRegression(max_iter=i)
        lr_model.fit(X_train, y_train)

        y_train_pred = lr_model.predict(X_train)
        y_test_pred = lr_model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    # Plotting the results
    # Used AI's help to plot the results same as the pdf file.
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 26), train_accuracies, label="Train Accuracy", marker='o')
    plt.plot(range(1, 26), test_accuracies, label="Test Accuracy", marker='s')
    plt.xlabel("max_iter")
    plt.ylabel("Accuracy")
    plt.title("Logistic Regression Accuracy vs. Iterations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main function to run the code
def main():
    train_df, test_df = prepare_dataset()

    scores = bias_scores(train_df)

    print(scores[:2])
    print(scores[-2:])

    X_train, X_test, y_train, y_test = vectorize_text(scores, train_df, test_df)
    train(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
