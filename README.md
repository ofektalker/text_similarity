# text_similarity

Keras logistic regression model that trains a neural network to learn to predict the most similar text sentence given a query.

## Input format

similarity.py --query <query file global path> --text <text file global path> --task <train/test> --data <training dataset in .csv format> --model <trained model>

### For training

similarity.py --task train --data <training dataset in .csv format> --model <model target file>

### For testing

similarity.py --query <query file global path> --text <text file global path> --task test --model <trained model>

as a test result a file named "most_similar.txt" will be created with the sentence most similar from text file to query text.

- used https://fasttext.cc/docs/en/english-vectors.html -> wiki-news-300d-1M.vec.zip - pre trained word vectors trained using fast text on the english language.
- used https://www.kaggle.com/shineucc/bbc-news-dataset as dataset
