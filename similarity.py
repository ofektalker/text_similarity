
import io
import argparse

import pandas as pd
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.metrics import cosine_similarity
from sklearn.model_selection import train_test_split

import os
import string
from datetime import datetime

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer


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
        final_result = list(map(lambda sentence: list(filter(lambda word: len(word) > 1, sentence)), stemmed_words))
    else:
        final_result = list(filter(lambda word: len(word) > 1, filtered_words))
    # All words after preprocessing
    return final_result


def load_vectors(fname):
    print(datetime.now(), ' --- Loading Vectors')
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    print(datetime.now(), ' --- Finished Loading Vectors')
    return data


WORD_VEC = load_vectors('wiki-news-300d-1M.vec')
STEM = False

####################################################################################
# ### CONFIGURE PROGRAM INPUT ### #
####################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--query',
                    metavar='Query',
                    type=str,
                    help='File containing query sentence.')
parser.add_argument('--text',
                    metavar='Text',
                    type=str,
                    help='File containing text to match query sentence')
parser.add_argument('--task',
                    metavar='Task',
                    choices=['train', 'test'],
                    type=str, help='Choose between train / test')
parser.add_argument('--data',
                    metavar='Data',
                    type=str,
                    help='Training dataset in CSV format.')
parser.add_argument('--model',
                    metavar='Model',
                    type=str,
                    help='Trained model')
args = parser.parse_args()


####################################################################################
####################################################################################

# expects dataset_filename to be the csv filename to load as dataset
# expects dataset to have 2 columns, 1 for data, 1 for queries
# returns [x, y] where x = data, y = queries
def load_from_data_set(dataset_filename):
    global args
    if type(dataset_filename) != str:
        print(datetime.now(), ' --- ERROR in function load_from_data_set: '
                              'argument \'dataset_filename\' is expected to be a string. '
                              'recieved ' + str(type(dataset_filename)))
        return None
    if dataset_filename[-4:] != '.csv':
        print(datetime.now(), ' --- ERROR in function load_from_data_set: '
                              'Expects file to be csv, recieved \'' + dataset_filename + '\'')
        return None
    file_path = os.path.join(os.getcwd(), dataset_filename)
    print(datetime.now(), ' --- Loading dataset.')
    # create header for dataset
    try:
        # read the dataset
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(datetime.now(), ' --- ERROR in load_from_data_set: File ' + file_path + ' does not exist.')
        return None

    df = df.replace('?', np.nan).replace('', np.nan)
    # drop the NaN
    df = df.dropna(axis=0, how="any")
    x, y = df.iloc[:, 1].values.tolist(), df.iloc[:, 2].values.tolist()
    print(datetime.now(), ' --- Dataset is loaded.')
    return [x, y]


def load_from_data_txt(data_filename, delim):
    print(datetime.now(), ' --- Loading ' + data_filename)
    file_path = os.path.join(os.getcwd(), data_filename)
    try:
        with open(file_path) as file:
            data = file.read()
    except FileNotFoundError:
        print(datetime.now(), ' --- ERROR in load_from_data_txt: File ' + file_path + ' does not exist.')
        return None

    if delim is not None:
        data = data.split(delim)
    print(datetime.now(), ' --- Loaded ' + data_filename)
    return data


def preprocess_list(data):
    print(datetime.now(), ' --- Preprocessing list.')
    if type(data) is not list:
        print(datetime.now(), ' --- ERROR in preprocess_list: data argument is not of type list, recieved ' +
              str(type(data)))
        return None
    result = [preprocess(d, STEM) for d in data]
    if None in result:
        return None
    print(datetime.now(), ' --- Preprocessed list.')
    return result


def sentence_embed(sentence):
    global WORD_VEC
    if WORD_VEC is None:
        print(datetime.now(), ' --- ERROR in sentence_embed: WORD_VEC is undefined')
        return None
    if type(sentence) is not list:
        print(datetime.now(), ' --- ERROR in sentence_embed: sentence expects a list, recieved ' + str(type(sentence)))
        return None
    sentence = list(filter(lambda word: word in WORD_VEC.keys(), sentence))
    sentence = list(map(lambda word: WORD_VEC[word], sentence))
    return sentence


def average(sentence):
    return [dimension / len(sentence) for dimension in np.sum(np.asarray(sentence), 0)]


def split_data(data, ratio):
    print(datetime.now(), ' --- Split Data into test and train.')
    x, y = data
    x_train, x_test, \
        y_train, y_test = train_test_split(x, y, test_size=ratio)
    print(datetime.now(), ' --- Data is split ' + str(ratio) + ' test / ' + str(1 - ratio) + ' train')
    return [x_train, x_test, y_train, y_test]


def prepare_data(data):
    print(datetime.now(), ' --- Preparing data as correct input for neural network.')
    result = np.asarray(data, dtype=np.float32)
    print(datetime.now(), ' --- Data is in correct format input for neural network.')
    return result


def construct_neural_network():
    layers, io_dim = 8, 300
    print(datetime.now(), ' --- Construct neural network.')
    model = Sequential()
    current_dim = int(io_dim - (io_dim / layers))
    model.add(Dense(current_dim, input_dim=io_dim, activation='relu'))
    for i in range(layers - 2):
        if current_dim > io_dim / 2:
            current_dim -= int(io_dim / layers)
        else:
            current_dim += int(io_dim / layers)
        model.add(Dense(current_dim, activation='relu'))
    model.add(Dense(io_dim))
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    print(datetime.now(), ' --- Neural Network ready.')
    return model


def train_model(model, x, y, epochs):
    print(datetime.now(), ' --- Training model')
    model.fit(x, y, epochs=epochs)
    print(datetime.now(), ' --- Finished Training model')
    return model


def evaluate_model(model, x, y, batch_size, verbose):
    score, accuracy = model.evaluate(x, y, batch_size=batch_size, verbose=verbose)
    print("Test fraction correct (NN-Score) = {:.2f}".format(score))
    print("Test fraction correct (NN-Accuracy) = {:.2f}".format(accuracy))


def save_model(model):
    print(datetime.now(), ' --- Saving Model to disk')
    model.save("final_model.h5")
    print(datetime.now(), ' --- Saved Model to disk')


def load_whole_model(model_filename):
    print(datetime.now(), ' --- Loading model from disk')
    model = construct_neural_network()
    model.load_weights(os.path.join(os.getcwd(), model_filename))
    print(datetime.now(), ' --- Loaded model from disk')
    return model


def apply_func_on_members(func, data):
    print(datetime.now(), ' --- Applying ' + func.__name__ + ' To data')
    result = [func(x) for x in data]
    print(datetime.now(), ' --- Applied ' + func.__name__)
    return result


def save_most_similar(s, i):
    print(datetime.now(), ' --- Saving prediction to file')
    with open(os.path.join(os.getcwd(), 'most_similar.txt'), 'w') as most_similar_file:
        most_similar_file.write(s[i])
    print(datetime.now(), ' --- Saved predition to \'most_similar.txt\'')


def feed_to_nn(query, data, model):
    print(datetime.now(), ' --- Feed data into model')
    prediction = model.predict(query)
    scores = cosine_similarity(prediction, data)
    coords_max = np.where(scores == np.amax(scores))
    list_of_coordinates = list(zip(coords_max[0], coords_max[1]))
    print(datetime.now(), ' --- Acquired prediction')
    return list_of_coordinates[0][1]


def train(dataset_filename):
    global WORD_VEC
    global STEM

    x, y = load_from_data_set(dataset_filename)
    if x is None or y is None:
        return

    x, y = preprocess_list(x), preprocess_list(y)
    if x is None or y is None or None in x or None in y:
        return

    x, y = apply_func_on_members(sentence_embed, x), apply_func_on_members(sentence_embed, y)

    x, y = apply_func_on_members(average, x), apply_func_on_members(average, y)

    x_train, x_test, y_train, y_test = split_data([x, y], 0.3)

    x_train, y_train = prepare_data(x_train), prepare_data(y_train)

    model = construct_neural_network()
    train_model(model, x_train, y_train, 100)
    evaluate_model(model, x_train, y_train, 16, 0)
    save_model(model)


def test(query_filename, text_filename, model_filename):
    global WORD_VEC
    global STEM
    model = load_whole_model(model_filename)

    x = load_from_data_txt(query_filename, None)
    y = load_from_data_txt(text_filename, '\n')
    o = y

    x, y = preprocess(x, STEM), preprocess_list(y)

    x, y = sentence_embed(x), apply_func_on_members(sentence_embed, y)

    x, y = average(x), apply_func_on_members(average, y)

    # sentences, query_data = prepare_data(sentences), prepare_data(query_data)[0]
    x = np.expand_dims(np.asarray(x, dtype=np.float32), 0)
    y = np.expand_dims(np.asarray(y, dtype=np.float32), 0)

    most_similar_index = feed_to_nn(x, y, model)

    save_most_similar(o, most_similar_index)


if args.task is not None:
    if args.task == 'train':
        train(args.data)
    elif args.task == 'test':
        if args.query is not None:
            if args.text is not None:
                if args.model is not None:
                    test(args.query, args.text, args.model)
                else:
                    print('please supply a model using \'--model <model_filename>.h5\'')
            else:
                print('please supply a text file using \'--text <text_filename>.txt\'')
        else:
            print('please supply a query file using \'--query <query_filname>.txt\'')
else:
    print('please use \'--task <task>\'\n<task> is any of the values: (train, test)')
