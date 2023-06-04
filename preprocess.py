import tensorflow as tf
import numpy as np
import re

def get_data(file_name):
    """
    Helpful documentation
    """
    train = []
    vocab = {}
    vocab_size = 0

    with open(file_name, "r") as file:
        for line in file:
            line = re.sub("[^\w\s]+", "", line)
            tokens = line.split()
            train.extend(tokens)
            
            for t in tokens:
                if t not in vocab:
                    vocab[t] = vocab_size
                    vocab_size += 1
    vocab['UNK'] = vocab_size
    vocab_size += 1

    # char2index = {char: index for index, char in enumerate(vocab)}
    # index2char = np.array(vocab)

    # text = open(file_name, mode='r').read()

    train = list(map(lambda x: vocab[x], train))
    # text_as_int = np.array([char2index[char] for char in text])


    # print(len(train))
    print(train[:15])
    return train, vocab


get_data("data/DonQ1.txt")
