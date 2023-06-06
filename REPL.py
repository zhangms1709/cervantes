import tensorflow as tf
import numpy as np
from preprocess import get_data

dataset, vocab = get_data("data/DonQ2.txt")
char2index = {char: index for index, char in enumerate(vocab)}
index2char = np.array(vocab)

model = tf.keras.models.load_model('quijote_rnn2.h5')

def generate_text(model, start_string, num_generate = 150, temperature=1.0):
    input_indices = [char2index[s] for s in start_string]
    input_indices = tf.expand_dims(input_indices, 0)

    text_generated = []

    model.reset_states()
    for char_index in range(num_generate):
        predictions = model(input_indices)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
        predictions,
        num_samples=1
        )[-1,0].numpy()

        input_indices = tf.expand_dims([predicted_id], 0)

        text_generated.append(index2char[predicted_id])

    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"don Quijote dice"))