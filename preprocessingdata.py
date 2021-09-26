# Loading in all of the dependecies that are required for this script
import pandas as pd
import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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


summary = review_dataset["Summary"]
text = review_dataset["Text"]


# A function that will do all of the preprocessing that we need to do for our data
def preprocessing_text(dataset, empty_dataset):
    for sentence in dataset:
        tokenized_sentence = word_tokenize(sentence)

        clean_sentence = []
        for word in tokenized_sentence:
            # removing non-alphanumeric elements from our text
            if word.isalnum() == False and word != ' ':
                tokenized_sentence.remove(word)

        for word in tokenized_sentence:
            # removing stop words from our text
            if word.lower() in stop_words:
                tokenized_sentence.remove(word)

        clean_sentence = []
        reviews_sentence = []
        new_sentence = []

        for sentence in input_data:
            for word in sentence:
                for character in word:
                    if character not in punctuation:
                        word = re.sub(r'[^\w\s]', '', word)
                        new_sentence.append(word)

            clean_sentence = ''.join(new_sentence)
        
        for word in clean_sentence:
            # performing stemming
            
            snowball_stemmer = SnowballStemmer('english')    
            # Using the Snowball Stemmer algorithm on each of the words of the updated list
            word = snowball_stemmer.stem(word)
            reviews_sentence.append(word)
            sentence = ''.join(clean_sentence)

        # lowercasing the text
        sentence = sentence.lower()

        # adding all of the updated values to the final list
        empty_dataset.append(sentence)
 
preprocessing_text(review_text, cleaned_text)
preprocessing_text(review_summary, cleaned_summary)

# displaying the updated reviews and summaries           
print(cleaned_text)
print(cleaned_summary)
