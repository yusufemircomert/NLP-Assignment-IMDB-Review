# Necessary imports
from datasets import load_dataset
import pandas as pd
import re
import string
from nltk.corpus import stopwords
import nltk
import math
from collections import Counter
from sklearn.metrics import accuracy_score




# It is a function to preprocess the text data.
def preprocess_text(text, stop_words): # I used AI help for this function
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

def prepare_dataset(stop_words):
    
    dataset = load_dataset("imdb")
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    train_df["text"] = train_df["text"].apply(lambda x: preprocess_text(x, stop_words))
    test_df["text"] = test_df["text"].apply(lambda x: preprocess_text(x, stop_words))
    print("train_df and test_df shapes.")
    print(train_df.shape)  # (25000, 2)
    print(test_df.shape)   # (25000, 2)
    print()
    return train_df, test_df

# Naive Bayes Classifier class
class NaiveBayesClassifier:
    def __init__(self):
        self.total_pos_words = 0
        self.total_neg_words = 0
        self.vocab_size = 0
        self.prior_pos = 0.0
        self.prior_neg = 0.0
        self.pos_counter = Counter()
        self.neg_counter = Counter()
        self.vocab = set()

    def fit(self, train_df):
        # Seperate positive and negative samples
        pos_texts = train_df[train_df["label"] == 1]["text"]
        neg_texts = train_df[train_df["label"] == 0]["text"]

        # Count the frequencies of each word in positive and negative samples
        for text in pos_texts:
            words = text.split()
            self.pos_counter.update(words)
            self.vocab.update(words)
            self.total_pos_words += len(words)

        # Count the frequencies of each word in negative samples
        for text in neg_texts:
            words = text.split()
            self.neg_counter.update(words)
            self.vocab.update(words)
            self.total_neg_words += len(words)

        # Vocab size
        self.vocab_size = len(self.vocab)

        # Priors
        total_samples = len(train_df)
        self.prior_pos = len(pos_texts) / total_samples
        self.prior_neg = len(neg_texts) / total_samples

    def predict(self, text):
        # Preprocessing
        processed = preprocess_text(text, stop_words=set(stopwords.words("english")))
        words = processed.split()

        # Priors
        log_prob_pos = math.log(self.prior_pos)
        log_prob_neg = math.log(self.prior_neg)

        # Calculate log probabilities for each word
        # Using laplace smoothing
        for word in words:
            # P(word | pos)
            word_freq_pos = self.pos_counter.get(word, 0) + 1
            prob_word_pos = word_freq_pos / (self.total_pos_words + self.vocab_size)
            log_prob_pos += math.log(prob_word_pos)

            # P(word | neg)
            word_freq_neg = self.neg_counter.get(word, 0) + 1
            prob_word_neg = word_freq_neg / (self.total_neg_words + self.vocab_size)
            log_prob_neg += math.log(prob_word_neg)

        # prediction
        y_predicted = 1 if log_prob_pos > log_prob_neg else 0
        
        return (y_predicted,log_prob_pos,log_prob_neg)

def making_tests(nb, test_df):
    #First example of the test: Example 1

    print(nb.total_pos_words)
    print(nb.total_neg_words)
    print(nb.vocab_size)
    print(nb.prior_pos)
    print(nb.prior_neg)
    print(nb.pos_counter["great"])
    print(nb.neg_counter["great"])
    print()

    # Examples : Example 2
    prediction1 = nb.predict(test_df.iloc[0]["text"])
    prediction2 = nb.predict("This movie will be place at 1st in my favourite movies!")
    prediction3 = nb.predict("I couldn't wait for the movie to end, so I, turned it off halfway through. :D It was a complete disappointment.")

    # Examples : Example 3
    print(f"{'Positive' if prediction1[0] == 1 else 'Negative'}")
    print(prediction1)

    print(f"{'Positive' if prediction2[0] == 1 else 'Negative'}")
    print(prediction2)

    print(f"{'Positive' if prediction3[0] == 1 else 'Negative'}")
    print(prediction3)
    print()
    
    # Accuracy calculation
    y_true = test_df["label"].values
    y_pred = [nb.predict(text)[0] for text in test_df["text"]]
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.5f}")

# Main function
# for running the code
def main():
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

    train_df, test_df = prepare_dataset(stop_words)

    print("Preprocessing examples:")
    print(preprocess_text("This movie will be place at 1st in my favourite, movies!", stop_words))
    print(preprocess_text("I couldn't wait for the movie to end, so I turned, it off halfway through. :D It was a complete disappointment.", stop_words))
    print()

    nb = NaiveBayesClassifier()
    nb.fit(train_df)

    making_tests(nb, test_df)

if __name__ == "__main__":
    main()