import tensorflow as tf
import numpy as np
import re

def split_input_target(chunk):
    """
    For every sequence, duplicate and shift it to create the input 
    and target text.
    """
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def get_data(file_name):
    """
    Cleans up and prepares data for model training using regexes. 
    Splits the data into training sequences and batches.
    """
    seq_length = 150
    batch_size = 64
    buffer_size = 10000

    text = open(file_name, mode='r', encoding='utf-8-sig').read()
    text = re.sub("[^\w\s]+", "", text)

    vocab = sorted(set(text))
    vocab_size = len(vocab)
    char2index = {char: index for index, char in enumerate(vocab)}

    text_as_int = np.array([char2index[char] for char in text])

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    return dataset, vocab
    # train = []
    # vocab = {}
    # vocab_size = 0
    # seq_length = 150
    # BATCH_SIZE = 64
    # BUFFER_SIZE = 10000

    # text = open(file_name, mode='r', encoding='utf-8-sig').read()

    # for line in text:
    #     line = re.sub("[^\w\s]+", "", line)
    #     tokens = line.split()
    #     train.extend(tokens)
            
    #     for t in tokens:
    #         if t not in vocab:
    #             vocab[t] = vocab_size
    #             vocab_size += 1
    # vocab['UNK'] = vocab_size
    # vocab_size += 1

    # char2index = {char: index for index, char in enumerate(vocab)}

    # text = open(file_name, mode='r', encoding='utf-8-sig').read()

    # # train = list(map(lambda x: vocab[x], train))
    # text_as_int = np.array([char2index.get(char, 'UNK') for char in text])
    # char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    # sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    # dataset = sequences.map(split_input_target)
    # dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    # # print(len(train))
    # # print(train[:15])
    # return dataset, vocab


# get_data("data/DonQ1.txt")
