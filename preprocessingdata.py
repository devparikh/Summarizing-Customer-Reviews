# Loading in all of the dependecies that are required for this script
import pandas as pd
import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer, lancaster

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

# Importing a list of all of the different stop words that we want to remove
stop_words = set(stopwords.words('english'))

# All of the different lists that we will put our cleaned data into
cleaned_summary = []
cleaned_text = []
cleaned_sentence = []

# A function that will do all of the preprocessing that we need to do for our data
def preprocessing_text(dataset, empty_sentence, empty_dataset):
    for sentence in dataset:
        empty_sentence = word_tokenize(sentence)
        empty_dataset.append(empty_sentence)

        for word in sentence:
            # removing non-alphabetic elements from our text
            if word.isalpha() == False:
                empty_sentence.remove(word)

            # removing stop words from our text
            if word.lower() in stop_words:
                empty_sentence.remove(word) 

            # removing duplicate elements from our list
            if empty_sentence[empty_sentence.index(word)] == empty_sentence[empty_sentence.index(word)-1]:
                # here the reason why we are converting to a dictionary is because dictionaries automatically remove all duplicates and so we converted back from dictionaries to lists
                empty_sentence = list(dict.fromkeys(empty_sentence))
        
        # performing stemmming

        lancaster = LancasterStemmer()    
        # Using the Lancaster Stemmer algorithm on each of the words of the updated list
        new_sentence = []
        for word in empty_sentence:
            word = lancaster.stem(word)
            new_sentence.append(word)
            new_sentence = ''.join(new_sentence)

        # lowercasing the text
        new_sentence = new_sentence.lower()

        # adding all of the updated values to the final list
        empty_dataset.append(new_sentence)

preprocessing_text(review_text, cleaned_sentence, cleaned_text)
preprocessing_text(review_summary, cleaned_sentence, cleaned_summary)

# displaying the updated reviews and summaries           
print(cleaned_text)
print(cleaned_summary)

