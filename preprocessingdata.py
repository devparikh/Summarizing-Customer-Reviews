# Loading in all of the dependecies that are required for this script
import pandas as pd
import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

nltk.download("all")

# Loading in the data
review_data = pd.read_csv("C:\\Users\\me\\Documents\\Text-Summarization\\Reviews.csv", index_col=0)

# Load in the data from the CSV file and then get the Summary and Text Column from the table
print(review_data.head())

# Analyzing some of the data from our dataset
print(review_data["Summary"].head())
print(review_data["Text"].head())

'''Data Cleaning'''

# Here we are removing any potential duplicate review
review_data.drop_duplicates(keep="first", inplace=True)

# drop all of the rows that have a NULL or NAN value
review_dataset = review_data.dropna()

'''Data Preprocessing'''

review_summary = []
sentences = []

for sentence in review_dataset["Summary"]:
    sentences = sentence
    if sentences != [""]:
        review_summary.append(sentences)

review_text = []
sentences = []

for sentence in review_dataset["Text"]:
    sentences = sentence
    if sentences != [""]:
        review_text.append(sentences)


# Removing duplicates
clean_reviews_text = []

for sentence in review_text:
    # we split the string into a list of words so that it is easier for us to be able to remove the duplicate that we want to
    split_sentence = sentence.split()
    for word in split_sentence:
        if split_sentence[split_sentence.index(word)] == split_sentence[split_sentence.index(word)-1]:
            # here the reason why we are converting to a dictionary is because dictionaries automatically remove all duplicates and so we converted back from dictionaries to lists
            split_sentence = list(dict.fromkeys(split_sentence))
            # now we joined every single element of the split sentence to create a new string
            clean_sentence = ' '.join(split_sentence)
            clean_reviews_text.append(clean_sentence)

# Here we are just performing tokenization on our data
def tokenize_text(dataset):
    for sentence in dataset:
        sentence = word_tokenize(sentence)

tokenize_text(clean_reviews_text)
tokenize_text(review_summary)

# removing non-alphabet tokens which will include special characters
non_alpha_reviews_summary = []
non_alpha_reviews_text = []

def removing_non_alphabetic_letters(dataset, empty_dataset):
    removing_non_alphabetic_letters = []
    for sentences in dataset:
        for word in sentences:
            if word.isalpha() == True:
                removing_non_alphabetic_letters.append(word)
        empty_dataset.append(removing_non_alphabetic_letters) 

removing_non_alphabetic_letters(clean_reviews_text,  non_alpha_reviews_text)     
removing_non_alphabetic_letters(review_summary, non_alpha_reviews_summary)  


removed_stopwords_summary = []
removed_stopwords_text = []

# Importing a list of all of the different stop words that we want to remove
stop_words = set(stopwords.words('english'))

def removing_stop_words(dataset, empty_dataset):
    removing_stop_words = []
    for sentence in dataset:
        for word in sentence:
            if word.lower() not in stop_words:
                removing_stop_words.append(word)
            empty_dataset.append(removing_stop_words)

removing_stop_words(non_alpha_reviews_text, removed_stopwords_text)
removing_stop_words(non_alpha_reviews_summary, removed_stopwords_summary)

cleaned_reviews_text = []
cleaned_reviews_summary = []

def snowball_stemming(dataset, empty_dataset):
    snowball_sentence = []
    for word in dataset:
        # performing stemmming
        snowball = SnowballStemmer('english')    
        # Using the Snowball Stemmer algorithm on each of the words of the updated list
        word = snowball.stem(word)
        snowball_sentence.append(word)
        concatanated_sentence = ''.join(snowball_sentence)
    empty_dataset.append(concatanated_sentence)

snowball_stemming(removed_stopwords_text, cleaned_reviews_text)
snowball_stemming(removed_stopwords_summary, cleaned_reviews_summary)

def lowercase_dataset(dataset):
    # here I converted everything into lowercase so that the computer does not think that HI and hi is a different word
    for sentence in dataset:
        sentence.lower()

lowercase_dataset(removed_stopwords_text)
lowercase_dataset(removed_stopwords_summary)

