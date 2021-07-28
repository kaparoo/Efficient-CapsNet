# -*- coding: utf-8 -*-

from absl import app
from absl import flags

import efficient_capsnet
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_dir",
                    None,
                    "Directory to save training results.",
                    required=True)
flags.DEFINE_integer("num_epochs",
                     3,
                     "Number of epochs.",
                     lower_bound=0,
                     upper_bound=100)
flags.DEFINE_float("validation_split",
                   0.2,
                   "Ratio for a validation dataset from training dataset.",
                   lower_bound=0.0,
                   upper_bound=0.5)
flags.DEFINE_boolean("show_score", False,
                     "Flag for scoring the trained model.")
flags.DEFINE_boolean("show_summary", False,
                     "Flag for displaying the model summary.")
flags.DEFINE_boolean("scale_mnist", False,
                     "Flag for scaling the MNIST dataset.")
flags.DEFINE_boolean("plot_logs", False, "Flag for plotting the saved logs.")


def _get_mnist_dataset(num_classes: int = 10, scaling: bool = False):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = tf.cast(tf.expand_dims(X_train, axis=-1), dtype=tf.float32)
    y_train = tf.one_hot(y_train, depth=num_classes, dtype=tf.float32)
    X_test = tf.cast(tf.expand_dims(X_test, axis=-1), dtype=tf.float32)
    y_test = tf.one_hot(y_test, depth=num_classes, dtype=tf.float32)

    if scaling is True:
        X_train = X_train / 255
        X_test = X_test / 255

    return (X_train, y_train), (X_test, y_test)


def _plot_training_logs(checkpoint_dir: str, dpi: int = 300) -> None:
    with open(f"{checkpoint_dir}/train_log.csv", mode='r') as csvfile:
        logs = np.array(
            [line.strip().split(',') for line in csvfile.readlines()])
        logs = logs[1:, :].astype(np.float)  # [1:,:]: remove header

        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(logs[:, 1], label="Training accuracy")
        plt.plot(logs[:, 3], label="Validation accuracy")
        plt.legend()
        plt.savefig(f"{checkpoint_dir}/accuracy.png", dpi=dpi)

        plt.clf()

        plt.title("Margin loss")
        plt.xlabel("Epoch")
        plt.ylabel("Margin loss")
        plt.plot(logs[:, 2], label="Training loss")
        plt.plot(logs[:, 4], label="Validation loss")
        plt.legend()
        plt.savefig(f"{checkpoint_dir}/loss.png", dpi=dpi)


def main(_) -> None:
    param = efficient_capsnet.make_param()
    model = efficient_capsnet.make_model(param)
    mnist_train, mnist_test = _get_mnist_dataset(param.num_digit_caps,
                                                 FLAGS.scale_mnist)
    X_train, y_train = mnist_train
    X_test, y_test = mnist_test

    checkpoint_dir = FLAGS.checkpoint_dir
    initial_epoch = 0
    num_epochs = FLAGS.num_epochs

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    else:
        checkpoints = [
            file for file in os.listdir(checkpoint_dir) if "ckpt" in file
        ]
        if len(checkpoints) != 0:
            checkpoints.sort()
            checkpoint_name = checkpoints[-1].split(".")[0]
            initial_epoch = int(checkpoint_name)
            model.load_weights(
                filepath=f"{checkpoint_dir}/{checkpoint_name}.ckpt")

    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=f"{checkpoint_dir}/train_log.csv", append=True)
    model_saver = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir +
                                                     "/{epoch:04d}.ckpt",
                                                     save_weights_only=True)
    model.fit(x=X_train,
              y=y_train,
              validation_split=FLAGS.validation_split,
              initial_epoch=initial_epoch,
              epochs=initial_epoch + num_epochs,
              callbacks=[csv_logger, model_saver])
    model.save(f"{checkpoint_dir}/model")
    param.save_config(f"{checkpoint_dir}/config.txt")

    if FLAGS.show_score is True:
        if initial_epoch == 0 and num_epochs == 0:
            print(f"ERROR")
        else:
            loss, score = model.evaluate(X_test, y_test)
            print(f"Test loss: {loss: .4f}")
            print(f"Test score: {score: .4f}")

    if FLAGS.show_summary is True:
        if initial_epoch == 0 and num_epochs == 0:
            print(f"ERROR")
        else:
            model.summary()

    if FLAGS.plot_logs is True:
        if initial_epoch == 0 and num_epochs == 0:
            print(f"ERROR")
        else:
            _plot_training_logs(checkpoint_dir, dpi=300)


if __name__ == "__main__":
    app.run(main)
