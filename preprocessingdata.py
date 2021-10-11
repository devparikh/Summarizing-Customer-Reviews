# Loading in all of the dependecies that are required for this script
import pandas as pd
import tensorflow as tf
import numpy as np
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
import re
from spellchecker import SpellChecker
import contractions
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

cleaned_text = []
cleaned_summary = []


# A function that will do all of the preprocessing that we need to do for our data
def preprocessing_text(dataset, empty_dataset):
    for sentence in dataset:
        # We are tokenizing the text
        # when using this tensorflow function we are automatically removing punctuation and other randon characters
        tokenizer = Tokenizer(num_words = 100, char_level=False, oov_token="UNK")
        tokenizer.fit_on_texts(sentence)
        # removing all word characters from our sentences
        sentence = re.sub(r"\W", " ", sentence, flags=re.I)
        split_sentence = sentence.split()
        
        # We are removing all non-alphanumeric values in sentences
        for word in split_sentence:
            for character in word:
                # removing non-alphanumeric elements from our text
                if word.isalnum() == False and word != ' ':
                    split_sentence.remove(word)
                    
        print("Successfully removed non-alphanumeric values")

        reviews = []
        # Here we are removing all stop words
        for word in split_sentence:
            # removing stop words from our text
            if word.lower() not in stop_words:
                reviews.append(word)
        print("Successfully removed stop words")

        clean_sentence = []
        new_sentence = []
        punctuation = '''"!#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''''
        # Here is the second part of the preprocessing were we are expanding contractions, correcting spellings, and removing random characters from text
        for word in reviews:
            # what we are going is that we are updating the word to fix the contractions from you've to your have so that the computer is able to understand it
            # we did it before the puntuation is removed that we hve can still make sense of this short forms that have characters in them
            for character in word:
                if character not in punctuation:
                    word = re.sub(r'[^\w\s]', '', word)

            word = contractions.fix(word)
            # Correcting the spelling of words
            spell = SpellChecker()

            spell.correction(word)

            def pos_tagger(nltk_tag):
                if nltk_tag.startswith('J'):
                    return wordnet.ADJ
                elif nltk_tag.startswith('V'):
                    return wordnet.VERB
                elif nltk_tag.startswith('N'):
                    return wordnet.NOUN
                elif nltk_tag.startswith('R'):
                    return wordnet.ADV
                else:         
                    return None

            pos_tagged = nltk.pos_tag(reviews)

            # Here what we are going is creating a list that maps the first index of each tuple from pos_tagged and then takes the second index or x[1] and puts that in the pos_tagger function to get a single letter pos tag that we can use 
            wordnet_tagger = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
            # performing lemmization using wordnet
            wordnetlemmatizer = WordNetLemmatizer()   
            for word, tag in wordnet_tagger:
                if tag is None:
                    new_sentence.append(word)
                else:
                    new_sentence.append(wordnetlemmatizer.lemmatize(word, tag))
           
        clean_sentence = ' '.join(new_sentence)
        
        # lowercasing the text
        clean_sentence = clean_sentence.lower()
        
        print(clean_sentence)
        # here we are taking these numeric values and then converting them into a sequence
        sequenced_text = tokenizer.texts_to_sequences(word_tokenize(clean_sentence))
        print(sequenced_text)
        print("converted text to a sequence of numeric values")
        # adding all of the updated values to the final list
        empty_dataset.append(clean_sentence)

preprocessing_text(review_text, cleaned_text)
preprocessing_text(review_summary, cleaned_summary)

# displaying the updated reviews and summaries           
print(cleaned_text)
print(cleaned_summary)
