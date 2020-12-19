import tensorflow as tf
import yaml
from pathlib import Path
from vanilla_gan import GAN
from data_gen import generate_sin_data

# Set up loss object and functions.
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def train(model, config):
    EPOCHS = 1000
    batch_size = config["vanilla-gan"]["batch_size"]
    for epoch in range(EPOCHS):
        data = generate_sin_data(batch_size)
        generator_loss_, discriminator_loss_ = model(data)

        if epoch % 100 == 0:
            print(f"Generator loss: {generator_loss_}")
            print(f"Discriminator loss: {discriminator_loss_}")


def main(config):
    model = GAN(config)
    train(model, config)


if __name__ == "__main__":
    config = yaml.load(Path("config.yml").read_text(), Loader=yaml.SafeLoader)
    main(config)

