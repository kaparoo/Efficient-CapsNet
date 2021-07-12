from absl import app
from absl import flags

from efficient_capsnet import EfficientCapsNet
from efficient_capsnet import CapsNetParam
from efficient_capsnet import MarginLoss

import matplotlib.pyplot as plt

import os

import tensorflow as tf
from tensorflow.keras.datasets import mnist

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_dir", None, "Directory to save weights.", required=True)
flags.DEFINE_integer("epochs", 3, "Number of epochs.", lower_bound=1, upper_bound=100)

def _get_mnist_dataset():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = tf.cast(tf.expand_dims(X_train, axis=-1), dtype=tf.float32)
    y_train = tf.one_hot(y_train, depth=10, dtype=tf.float32)
    X_test = tf.cast(tf.expand_dims(X_test, axis=-1), dtype=tf.float32)
    y_test = tf.one_hot(y_test, depth=10, dtype=tf.float32)
    return (X_train, y_train), (X_test, y_test)

def main(_):
    (X_train, y_train), _ = _get_mnist_dataset()
    param = CapsNetParam()
    model = EfficientCapsNet(param)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=MarginLoss(), metrics=["accuracy"])

    checkpoint_dir = FLAGS.checkpoint_dir
    if os.path.exists(checkpoint_dir):
        checkpoints = [name for name in os.listdir(FLAGS.checkpoint) if "ckpt" in name]
        checkpoints.sort()
        checkpoint_name = checkpoints[-1].split('.')[0]
        epoch = int(checkpoint_name) + 1
        model.load_weights(f"{checkpoint_dir}/{checkpoint_name}.ckpt")
    else:
        os.makedirs(checkpoint_dir)

    epoch = 1
    epochs = FLAGS.epochs
    history = None
    while epochs is None or epoch <= epochs:
        print(f"Epoch: {epoch}/{epochs}")
        history = model.fit(X_train, y_train, validation_split=0.2)
        model.param.save_config(f"{checkpoint_dir}/config.txt")
        model.save_weights(f"{checkpoint_dir}/{epoch:05d}.ckpt")
        model.save(f"{checkpoint_dir}/model")
        epoch += 1

    plt.plot(history.history["acc"])
    plt.plot(history.history["val_acc"])
    plt.title("Train accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"], loc="upper left")
    plt.show()


if __name__ == "__main__":
    app.run(main)