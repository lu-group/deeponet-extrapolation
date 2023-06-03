import os
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
import numpy as np
import deepxde as dde
from deepxde.backend import tf


def gelu(x):
    return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def low_fidelity_u():
    test_data = np.load(f"cavity_extrapolation.npz")
    X_test = (test_data["branch_extrapolation"], test_data["trunk_extrapolation"])
    u_test = test_data["u_extrapolation"].reshape(-1, 1)
    data = dde.data.Triple(X_train=X_test, y_train=u_test, X_test=X_test, y_test=u_test)

    net = dde.maps.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
    )
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])

    model.restore(f"model_u/model", verbose=1)
    u_pred = model.predict(X_test).reshape(10, 101*101)
    u_pred = np.repeat(u_pred, 10, axis=0)
    np.savetxt(f"low_fidelity_u_10times.dat", u_pred)


def low_fidelity_v():
    test_data = np.load(f"cavity_extrapolation.npz")
    X_test = (test_data["branch_extrapolation"], test_data["trunk_extrapolation"])
    v_test = test_data["v_extrapolation"].reshape(-1, 1)
    data = dde.data.Triple(X_train=X_test, y_train=v_test, X_test=X_test, y_test=v_test)

    net = dde.maps.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
    )
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])

    model.restore(f"model_v/model", verbose=1)
    v_pred = model.predict(X_test).reshape(10, 101*101)
    v_pred = np.repeat(v_pred, 10, axis=0)
    np.savetxt(f"low_fidelity_v_10times.dat", v_pred)


if __name__ == "__main__":
    dde.utils.apply(low_fidelity_u)
    dde.utils.apply(low_fidelity_v)

