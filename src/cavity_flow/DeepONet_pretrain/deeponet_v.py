import os
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
import numpy as np
import deepxde as dde
from deepxde.backend import tf


def gelu(x):
    return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def main():
    lr = 0.001
    iterations = 500000
    data_train = np.load(f"../../../data/cavity_flow/cavity_train.npz")
    data_test = np.load(f"../../../data/cavity_flow/cavity_test.npz")
    X_train = (data_train["branch_train"], data_train["trunk_train"])
    y_train = data_train["v_train"]
    X_test = (data_test["branch_test"], data_test["trunk_test"])
    y_test = data_test["v_test"]
    data = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    net = dde.maps.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
    )

    def loss_func(y_true, y_pred):
        return tf.math.reduce_mean(
            tf.norm(tf.reshape(y_true, [-1, 101 * 101]) - tf.reshape(y_pred, [-1, 101 * 101]), axis=1) / tf.norm(
                tf.reshape(y_true, [-1, 101 * 101]), axis=1))

    model = dde.Model(data, net)
    model.compile("adam", lr=lr, loss=loss_func, metrics=["l2 relative error"])

    checker = dde.callbacks.ModelCheckpoint("model_v/model.ckpt", save_better_only=False, period=10000)
    losshistory, train_state = model.train(iterations=iterations, callbacks=[checker], batch_size=102010)
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)


if __name__ == "__main__":
    main()
