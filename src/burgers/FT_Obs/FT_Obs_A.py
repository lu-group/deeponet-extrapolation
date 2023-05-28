import os
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
import numpy as np
import tensorflow.compat.v1 as tf
from multiprocessing import Pool


def apply(func, args=None, kwds=None):
    """
    Launch a new process to call the function.
    This can be used to clear Tensorflow GPU memory after model execution.
    """
    with Pool(1) as p:
        if args is None and kwds is None:
            r = p.apply(func)
        elif kwds is None:
            r = p.apply(func, args=args)
        elif args is None:
            r = p.apply(func, kwds=kwds)
        else:
            r = p.apply(func, args=args, kwds=kwds)
    return r


def gelu(x):
    return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def periodic(x):
    return tf.concat([tf.math.cos(x[:, 0:1] * 2 * np.pi), tf.math.sin(x[:, 0:1] * 2 * np.pi),
                      tf.math.cos(2 * x[:, 0:1] * 2 * np.pi), tf.math.sin(2 * x[:, 0:1] * 2 * np.pi),
                      tf.math.cos(3 * x[:, 0:1] * 2 * np.pi), tf.math.sin(3 * x[:, 0:1] * 2 * np.pi),
                      tf.math.cos(4 * x[:, 0:1] * 2 * np.pi), tf.math.sin(4 * x[:, 0:1] * 2 * np.pi), x[:, 1:2]], 1)


def ft_obs_a(repeat, ls_test, num_train):
    import deepxde as dde
    dde.optimizers.config.set_LBFGS_options(maxiter=1000)

    data_test = np.load(f"../../../data/burgers/burgers_test_ls_{ls_test}_num_{num_train}.npz")
    sensor_value = data_test['x_branch_test'][repeat]
    X_train = data_test['x_trunk_test_select'][repeat]
    y_train = data_test['y_test_select'][repeat]
    X_test = np.array([[a, b] for a in np.linspace(0, 1, 101) for b in np.linspace(0, 1, 101)])
    y_test = data_test['y_test'][repeat]
    data = dde.data.dataset.DataSet(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    net = dde.maps.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
        trainable_branch=True,
        trainable_trunk=[True, True, True],
    )

    net.apply_feature_transform(periodic)
    net.build()
    net.inputs = [sensor_value, None]

    model = dde.Model(data, net)
    model.compile("L-BFGS", metrics=["l2 relative error"])

    model.train(epochs=1000, display_every=100, model_restore_path=f"model/model")

    return model.predict(X_test).reshape(1, 101*101)


def main():
    ls_test = 0.6
    num_train = 100
    output_list = []
    for repeat in range(100):
        output = apply(ft_obs_a, (repeat, ls_test, num_train))
        output_list.append(output)
    output_list = np.concatenate(output_list, axis=0)
    np.save(f"predict_ft_obs_a.npy", output_list)


if __name__ == "__main__":
    main()