import tensorflow as tf
import matplotlib.pyplot as plt


def display_images(out, label=None, count=False):
    """Inspired by: https://github.com/Atcold/pytorch-Deep-Learning/blob/master/11-VAE.ipynb"""
    out_pic = tf.reshape(out, [-1, 28, 28])
    plt.figure(figsize=(18, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(out_pic[i + 4])
        plt.axis("off")
        if count:
            plt.title(str(4 + i), color="w")
    plt.show()
