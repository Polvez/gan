import tensorflow.keras.layers as tfkl
import tensorflow as tf


class Generator(tfkl.Layer):
    def __init__(self, latent_dim: int, original_dim: int):
        super(Generator, self).__init__()
        self.layer1 = tfkl.Dense(latent_dim)
        self.layer1_lrl = tfkl.LeakyReLU()
        self.layer2 = tfkl.Dense(latent_dim)
        self.layer2_lrl = tfkl.LeakyReLU()
        self.out = tfkl.Dense(original_dim)

    def call(self, x):
        x = self.layer1_lrl(self.layer1(x))
        x = self.layer2_lrl(self.layer2(x))
        output = self.out(x)

        return output


class Discriminator(tfkl.Layer):
    def __init__(self, latent_dim: int, original_dim: int):
        super(Discriminator, self).__init__()
        self.layer1 = tfkl.Dense(latent_dim)
        self.layer1_lrl = tfkl.LeakyReLU()
        self.layer2 = tfkl.Dense(latent_dim)
        self.layer2_lrl = tfkl.LeakyReLU()
        self.layer3 = tfkl.Dense(original_dim)
        self.out = tfkl.Dense(1)

    def call(self, x):
        x = self.layer1_lrl(self.layer1(x))
        x = self.layer2_lrl(self.layer2(x))
        x = self.layer3(x)
        output = self.out(x)

        return output


class GAN(tf.keras.Model):
    def __init__(self, config):
        super(GAN, self).__init__()
        self.config = config["vanilla-gan"]

        self.generator = Generator(
            latent_dim=self.config["latent_dim"],
            original_dim=self.config["original_dim"],
        )
        self.discriminator = Discriminator(
            latent_dim=self.config["latent_dim"],
            original_dim=self.config["original_dim"],
        )

        # Set up optimizers for both models.
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def discriminator_loss(self, actual_output, generated_output):
        real_loss = self.cross_entropy(tf.ones_like(actual_output), actual_output)
        generated_loss = self.cross_entropy(tf.zeros_like(generated_output), generated_output)
        total_loss = real_loss + generated_loss
        return total_loss

    def generator_loss(self, generated_output):
        return self.cross_entropy(tf.ones_like(generated_output), generated_output)

    def discriminator_train_step(self, x, noise):

        with tf.GradientTape() as discriminator_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(x, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            discriminator_loss = self.discriminator_loss(real_output, fake_output)

        discriminator_gradients = discriminator_tape.gradient(
            discriminator_loss, self.discriminator.trainable_variables
        )
        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator.trainable_variables)
        )
        return discriminator_loss

    def generate_sample(self):
        noise = tf.random.normal([self.config["batch_size"], self.config["noise_dim"]])
        generated_sample = self.generator(noise, training=True)
        return generated_sample

    def generator_train_step(self,  noise):

        with tf.GradientTape() as generator_tape:
            generated_samples = self.generator(noise, training=True)

            fake_output = self.discriminator(generated_samples, training=True)
            generator_loss = self.generator_loss(fake_output)

        generator_gradients = generator_tape.gradient(
            generator_loss, self.generator.trainable_variables
        )
        self.generator_optimizer.apply_gradients(
            zip(generator_gradients, self.generator.trainable_variables)
        )

        return generator_loss


    def call(self, x):
        noise = tf.random.normal([self.config["batch_size"], self.config["noise_dim"]])

        discriminator_loss = self.discriminator_train_step(x, noise)
        generator_loss = self.generator_train_step(noise)

        return discriminator_loss, generator_loss
