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
    Cleans up and reads data for model training using utf-8. 
    Splits the data into training sequences and batches.
    """
    seq_length = 150
    batch_size = 64
    buffer_size = 10000

    text = open(file_name, mode='r', encoding='utf-8-sig').read()

    vocab = sorted(set(text))
    char2index = {char: index for index, char in enumerate(vocab)}

    text_as_int = np.array([char2index[char] for char in text])

    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

    dataset = sequences.map(split_input_target)
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    return dataset, vocab
