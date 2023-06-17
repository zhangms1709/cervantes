import numpy as np

class DataGenerator:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length

    def load_data_from_file(self, file_name):
        with open(file_name, 'r') as file:
            text = file.read().lower().replace('\n', ' ')
        words = text.split()
        unique_words = sorted(list(set(words)))
        word_to_idx = {word: index for index, word in enumerate(unique_words)}
        idx_to_word = {index: word for index, word in enumerate(unique_words)}
        data = self.text_to_sequence(text, word_to_idx)
        return data, word_to_idx, idx_to_word

    def text_to_sequence(self, text, word_to_idx):
        words = text.split()
        sequence = [word_to_idx[word] for word in words if word in word_to_idx]
        num_batches = len(sequence) // self.sequence_length
        sequence = sequence[:num_batches * self.sequence_length]
        return np.array(sequence)

    def generate_batch(self, data, batch_index, batch_size):
        start_index = batch_index * batch_size
        end_index = start_index + batch_size * self.sequence_length
        batch_data = data[start_index:end_index]
        batch_data = np.reshape(batch_data, (batch_size, self.sequence_length))
        return batch_data

    def sequence_to_text(self, sequence, idx_to_word):
        words = [idx_to_word[idx] for idx in sequence]
        text = ' '.join(words)
        return text

