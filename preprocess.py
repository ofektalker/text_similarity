# Dean Abutbul 305622375
# Ofek Talker 311369961
import os
import string
from datetime import datetime

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter
from wordcloud import WordCloud
import itertools


# data: String - A string to preprocess
# sentence_tokenize: Boolean - should the data be split by sentences
# stem: Boolean - should the data be stemmed
# returns list of words if sentence_tokenize was false
# else returns list of sentences, which are lists of words
def preprocess(data, stem):
    if type(data) != str:
        print(datetime.now(), ' --- ERROR in function preprocess: argument \'data\' should be a string to preprocess.'
                              ' Instead recieved ' + str(type(data)) + '\n' + str(data))
        return None
    # Clean punctuation
    data.translate(str.maketrans('', '', string.punctuation))

    # Word Tokenizing
    words = word_tokenize(data)

    # Stop Words Removal
    filtered_words = [w for w in words if w not in stopwords.words('english')]

    if stem:
        # Stemming
        stemmer = PorterStemmer()
        stemmed_words = list(map(lambda sentence: [stemmer.stem(w) for w in sentence], filtered_words))

    if stem:
        final_result = list(map(lambda sentence: list(filter(lambda word: len(word) > 1, sentence)), stemmed_words))
    else:
        final_result = list(filter(lambda word: len(word) > 1, filtered_words))
    # All words after preprocessing
    return final_result


def produce_word_cloud(data, filename):

    # Calculate Word Frequencies
    freq_dic = dict(Counter(data))
    # Sort by frequency
    sorted_dic = dict(sorted(freq_dic.items(), key=lambda kv: kv[1], reverse=True))
    # Take only top 20
    top_20 = dict(itertools.islice(sorted_dic.items(), 20))
    # generate and save to file
    WC = WordCloud().generate_from_frequencies(top_20)
    WC.to_file(os.path.join(os.getcwd(), filename.replace('.txt', '_cloud.png')))


def one_hot_encode(data, filename):
    # Construct 1hot representation for each word
    num_of_words = len(data)
    index_dictionary = dict(map(lambda word: (word, [0 if i != data.index(
        word) else 1 for i in range(num_of_words - 1)]), data))

    # Replace each word in sentences with appropriate 1hot representation
    onehot = list(map(lambda word: index_dictionary[word], data))

    # Save result into file
    with open(os.path.join(os.getcwd(), filename.replace('.txt', '1hot.txt')), 'a') as file:
        file.write(str(onehot))


def lab1_task(filename):
    rmv_punctuation, stem = True, True

    with open(filename, 'r') as file:
        data = file.read().replace('\n', ' ')

    clean_data = preprocess(data, rmv_punctuation, stem)
    produce_word_cloud(clean_data, filename)
    one_hot_encode(clean_data, filename)
