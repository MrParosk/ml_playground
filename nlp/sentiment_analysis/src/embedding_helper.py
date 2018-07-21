import os
import numpy as np


def create_embeddings_index(data_dir, embedding_file):
    """
    Creating a dictionary for words and their word-embeddings (based on pre-trained ones)
    Arguments:
        data_dir -- direction to the data folder
        embedding_file -- name of the embedded vectors

    Returns:
    embeddings_index -- dictionary for words and their word-embeddings
    """

    # Making sure that the embedding data exist
    if not os.path.exists(data_dir + embedding_file):
        raise Exception("You haven't downloaded the embedded file")

    embeddings_index = {}
    with open(os.path.join(data_dir, embedding_file), encoding='utf-8') as glove_file:
        for row in glove_file:
            values = row.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    return embeddings_index


def create_embedding_matrix(embeddings_index, word_index, num_words, embedding_dim):
    """
    Creating an embeddings matrix
    Arguments:
        embeddings_index -- dictionary for words and their word-embeddings
        word_index -- dictionary for words and their index
        num_words -- number of words to include in the embedded matrix
        embedding_dim -- the dimension of the embeddings

    Returns:
        embedding_matrix -- an embeddings matrix
    """

    embedding_matrix = np.zeros((num_words, embedding_dim))

    for word, i in word_index.items():
        if i >= num_words:
            continue

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # All words not found in embedding index will be zeros
            embedding_matrix[i] = embedding_vector

    return embedding_matrix
