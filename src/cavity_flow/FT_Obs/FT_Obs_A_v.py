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


def ft_obs_a(repeat, num_train):
    import deepxde as dde
    dde.optimizers.config.set_LBFGS_options(maxiter=1000)

    data_test = np.load(f"../../../data/cavity_flow/cavity_extrapolation_num_{num_train}_10times.npz")
    sensor_value = data_test['x_branch_test'][repeat]
    X_train = data_test['x_trunk_test_select'][repeat]
    y_train = data_test['v_test_select'][repeat]
    X_test = data_test['x_trunk_test'][repeat]
    y_test = data_test['v_test'][repeat]
    data = dde.data.dataset.DataSet(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    net = dde.maps.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
        trainable_branch=True,
        trainable_trunk=[True, True, True],
    )

    net.build()
    net.inputs = [sensor_value, None]

    model = dde.Model(data, net)
    model.compile("L-BFGS", metrics=["l2 relative error"])

    model.train(epochs=1000, display_every=100, model_restore_path=f"model_v/model")

    return model.predict(X_test).reshape(1, 101*101)


def main():
    num_train = 100
    output_list = []
    for repeat in range(100):
        output = apply(ft_obs_a, (repeat, num_train))
        output_list.append(output)
    output_list = np.concatenate(output_list, axis=0)
    np.save(f"predict_ft_obs_a_v.npy", output_list)


if __name__ == "__main__":
    main()
