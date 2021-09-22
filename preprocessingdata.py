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

# All of the different lists that we will put our cleaned data into
cleaned_summary = []
cleaned_text = []

stop_words = set(stopwords.words('english'))

# A function that will do all of the preprocessing that we need to do for our data
def preprocessing_text(dataset, empty_dataset):
    for sentence in dataset:
        tokenized_sentence = word_tokenize(sentence)

        new_sentence = []
        for word in tokenized_sentence:
             # removing non-alphabetic elements from our text
            if word.isalpha() == False:
                tokenized_sentence.remove(word)

        for word in tokenized_sentence:
            # removing stop words from our text
            if word.lower() in stop_words:
                tokenized_sentence.remove(word) 
        
        # the reason why we use a second loop is because the first one is looping over the empty_sentence when it is not cleaned but if we loop over it after it has been updated through a new loop the problem of characters not being in the list will be gone
        for word in tokenized_sentence:
            # removing duplicate elements from our list
            if tokenized_sentence[tokenized_sentence.index(word)] == tokenized_sentence[tokenized_sentence.index(word)-1]:
                # here the reason why we are converting to a dictionary is because dictionaries automatically remove all duplicates and so we converted back from dictionaries to lists
                cleaned_sentence = list(dict.fromkeys(tokenized_sentence))
            
            # performing stemmming
            snowball = SnowballStemmer('english')    
            # Using the Lancaster Stemmer algorithm on each of the words of the updated list
            word = snowball.stem(word)
            new_sentence.append(word)
            sentence = ''.join(new_sentence)

        # lowercasing the text
        sentence = sentence.lower()

        # adding all of the updated values to the final list
        empty_dataset.append(sentence)
 
preprocessing_text(review_text, cleaned_text)
preprocessing_text(review_summary, cleaned_summary)
