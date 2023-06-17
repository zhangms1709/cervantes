import tensorflow as tf
import numpy as np

class SeqGAN(tf.keras.Model):
    def __init__(self, sequence_length, vocab_size, latent_dim):
        super(SeqGAN, self).__init__()
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    def build_generator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, input_shape=(self.latent_dim,), activation='relu'),
            tf.keras.layers.Dense(self.sequence_length * self.vocab_size, activation='softmax'),
            tf.keras.layers.Reshape((self.sequence_length, self.vocab_size))
        ])
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences=False), input_shape=(self.sequence_length, self.vocab_size)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def call(self, inputs):
        return self.generator(inputs)

    def train_step(self, real_sequences):
        batch_size = tf.shape(real_sequences)[0]
        latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as tape:
            real_sequences = tf.reshape(real_sequences, shape=(-1, self.sequence_length))
            real_sequences = tf.one_hot(real_sequences, depth=self.vocab_size)
            real_sequences = tf.reshape(real_sequences, shape=(batch_size, self.sequence_length, self.vocab_size))
            
            fake_sequences = self.generator(latent_vectors)

            real_logits = self.discriminator(real_sequences)
            fake_logits = self.discriminator(fake_sequences)

            d_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

            # Gradient penalty for regularization
            alpha = tf.random.uniform(shape=[batch_size, 1, 1], minval=0.0, maxval=1.0)
            interpolated_sequences = alpha * real_sequences + (1 - alpha) * fake_sequences
            with tf.GradientTape() as penalty_tape:
                penalty_tape.watch(interpolated_sequences)
                interpolated_logits = self.discriminator(interpolated_sequences)
            gradients = penalty_tape.gradient(interpolated_logits, interpolated_sequences)
            gradient_norms = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
            gradient_penalty = tf.reduce_mean(tf.square(gradient_norms - 1))

            d_loss += 10 * gradient_penalty

            g_loss = -tf.reduce_mean(fake_logits)

        gradients = tape.gradient([g_loss, d_loss], self.generator.trainable_variables +
                                self.discriminator.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables +
                                        self.discriminator.trainable_variables))
        return g_loss, d_loss


    def generate_sequence(self):
        latent_vectors = tf.random.normal(shape=(1, self.latent_dim))
        generated_logits = self.generator(latent_vectors)
        generated_sequence = tf.argmax(generated_logits, axis=2)[0]
        return generated_sequence