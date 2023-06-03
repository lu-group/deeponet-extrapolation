import os
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
import numpy as np
import tensorflow as tf
from multiprocessing import Pool


def apply(func, args=None, kwds=None):
    """
    Launch a new process to call the function.
    This can be used to clear Tensorflow GPU memory after model execution:
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

    net = dde.nn.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
    )
    data_test = np.load(f"../../../data/cavity_flow/cavity_extrapolation_num_{num_train}_10times.npz")
    sensor_value = data_test['x_branch_test'][repeat]

    X_train_u_addition = np.tile(sensor_value, [num_train, 1])
    X_train_addition = data_test['x_trunk_test_select'][repeat]
    y_train_addition = data_test['u_test_select'][repeat]

    train_data = np.load("../../../data/cavity_flow/cavity_train.npz")
    # only choose k = 0.32, 0.34, 0.36, 0.38, 0.40 from DeepONet training dataset
    X_train = (np.concatenate([train_data["branch_train"][-5*10201:], X_train_u_addition], axis=0),
               np.concatenate([train_data["trunk_train"][-5*10201:], X_train_addition], axis=0))
    y_train = np.concatenate([train_data["u_train"][-5*10201:], y_train_addition], axis=0)
    X_test0 = np.tile(sensor_value, [10201, 1])
    X_test1 = data_test['x_trunk_test'][repeat]
    X_test = (X_test0, X_test1)
    y_test = data_test['u_test'][repeat]
    data = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    model = dde.Model(data, net)
    loss_func = lambda y_true, y_pred: dde.losses.mean_squared_error(
        y_true[:-100], y_pred[:-100]) + 0.3 * dde.losses.mean_squared_error(y_true[-100:], y_pred[-100:])
    iterations = 3000
    model.compile("adam", lr=0.0005, loss=loss_func, metrics=["l2 relative error"],
                  decay=("inverse time", iterations // 5, 0.8))
    model.train(iterations=iterations, display_every=100, model_restore_path=f"model_u/model")

    return model.predict(X_test).reshape(1, 101*101)


if __name__ == "__main__":
    num_train = 100
    output_list = []
    for repeat in range(100):
        output = apply(ft_obs_a, [repeat, num_train])
        output_list.append(output)
    output_list = np.concatenate(output_list, axis=0)
    np.save(f"predict_ft_obs_t_u.npy", output_list)