from model import SeqGAN
from datap import DataGenerator
import numpy as np
import tensorflow as tf

def train_gan(file_name, num_epochs, batch_size, sequence_length, latent_dim):
    data_generator = DataGenerator(sequence_length)
    data, word_to_idx, idx_to_word = data_generator.load_data_from_file(file_name)

    vocab_size = len(word_to_idx)
    model = SeqGAN(sequence_length, vocab_size, latent_dim)

    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(num_epochs):
        num_batches = len(data) // batch_size
        for batch_index in range(num_batches):
            batch_data = data_generator.generate_batch(data, batch_index, batch_size)

            with tf.GradientTape() as tape:
                g_loss, d_loss = model.train_step(batch_data)

            gradients = tape.gradient([g_loss, d_loss], model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if batch_index % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_index}/{num_batches}, "
                      f"Generator Loss: {g_loss:.4f}, Discriminator Loss: {d_loss:.4f}")

        # Generate a sample sequence after each epoch
        sample_sequence = model.generate_sequence()
        generated_text = data_generator.sequence_to_text(sample_sequence, idx_to_word)
        print(f"Generated Text: {generated_text}")
    model.save('seqgan_model')

if __name__ == '__main__':
    file_name = 'data/DonQ2.txt'  # Provide the name of your text file here
    num_epochs = 1
    batch_size = 64
    sequence_length = 20
    latent_dim = 50

    train_gan(file_name, num_epochs, batch_size, sequence_length, latent_dim)
