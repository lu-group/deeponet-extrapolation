import os
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
import numpy as np
import deepxde as dde
from deepxde.backend import tf


def gelu(x):
    return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def main():
    ls_train = 0.5
    ls_test = 0.5
    lr = 0.001
    epochs = 500000
    m_train = 101
    m_test = 101
    train_data = np.load(f"../../../data/diffusion_reaction/dr_train_ls_{ls_train}_{m_train}_{m_train}.npz")
    test_data = np.load(f"../../../data/diffusion_reaction/dr_test_ls_{ls_test}_{m_test}_{m_test}.npz")
    X_train = (np.repeat(train_data["X_train0"], m_train*m_train, axis=0), np.tile(train_data["X_train1"], (1000, 1)))
    y_train = train_data["y_train"].reshape(-1, 1)
    X_test = (np.repeat(test_data["X_test0"], m_test*m_test, axis=0), np.tile(test_data["X_test1"], (100, 1)))
    y_test = test_data["y_test"].reshape(-1, 1)
    data = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    net = dde.maps.DeepONet(
        [m_train, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
    )

    def dirichlet(inputs, outputs):
        x_trunk = inputs[1]
        x, t = x_trunk[:, 0:1], x_trunk[:, 1:2]
        return 10 * x * (1 - x) * t * (outputs + 1)
    net.apply_output_transform(dirichlet)

    model = dde.Model(data, net)
    model.compile("adam", lr=lr, metrics=["l2 relative error"], decay=("inverse time", epochs // 5, 0.5))

    checker = dde.callbacks.ModelCheckpoint("model/model.ckpt", save_better_only=False, period=1000)
    losshistory, train_state = model.train(epochs=epochs, callbacks=[checker], batch_size=30000)
    dde.saveplot(losshistory, train_state, issave=True, isplot=False)


if __name__ == "__main__":
    main()
