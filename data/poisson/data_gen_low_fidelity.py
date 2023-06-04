import os
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
import numpy as np
import deepxde as dde
from deepxde.backend import tf


def run(ls):
    def gelu(x):
        return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    test_data = np.load(f"poisson_test_ls_{ls}_5082.npz")
    X_test = (np.repeat(test_data["X_test0"], 5082, axis=0), np.tile(test_data["X_test1"], (100, 1)))
    y_test = test_data["y_test"].reshape(-1, 1)
    data = dde.data.Triple(X_train=X_test, y_train=y_test, X_test=X_test, y_test=y_test)

    net = dde.maps.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
    )

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])

    model.restore(f"model/model", verbose=1)
    y_pred = model.predict(X_test)
    np.savetxt(f"low_fidelity_{ls}.dat", y_pred.reshape(100, 5082))
    print(dde.metrics.l2_relative_error(y_test, y_pred))


if __name__ == "__main__":
    run(0.2)
