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


def dirichlet(inputs, outputs):
    x_trunk = inputs[1]
    x, t = x_trunk[:, 0:1], x_trunk[:, 1:2]
    return 4 * x * t * outputs + tf.sin(np.pi * x) + tf.sin(0.5 * np.pi * t)


def deeponet_together(repeat, ls_test, num_train):
    import deepxde as dde

    class Triple(dde.data.data.Data):
        def __init__(self, X_train, y_train, X_test, y_test):
            self.train_x = X_train
            self.train_y = y_train
            self.test_x = X_test
            self.test_y = y_test
            self.train_x_branch_obs = X_train[0][-num_train:]
            self.train_x_trunk_obs = X_train[1][-num_train:]
            self.train_y_obs = y_train[-num_train:]
            self.train_sampler = dde.data.sampler.BatchSampler(len(self.train_y) - num_train, shuffle=True)

        def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
            return loss_fn(targets, outputs)

        def train_next_batch(self, batch_size=None):
            if batch_size is None:
                return self.train_x, self.train_y
            indices = self.train_sampler.get_next(batch_size)
            return (
                (np.concatenate([self.train_x[0][indices], self.train_x_branch_obs]),
                 np.concatenate([self.train_x[1][indices], self.train_x_trunk_obs])),
                np.concatenate([self.train_y[indices], self.train_y_obs]),
            )

        def test(self):
            return self.test_x, self.test_y

    def loss_func(y_true, y_pred):
        return dde.losses.mean_squared_error(y_true[:-num_train], y_pred[:-num_train]) + \
               0.3 * dde.losses.mean_squared_error(y_true[-num_train:], y_pred[-num_train:])

    data_test = np.load(f"../../../data/advection/adv_test_ls_{ls_test}_num_{num_train}.npz")
    sensor_value = data_test['x_branch_test'][repeat]
    X_train_branch_addition = np.tile(sensor_value, [num_train, 1])
    X_train_trunk_addition = data_test['x_trunk_test_select'][repeat]
    y_train_addition = data_test['y_test_select'][repeat]
    X_train = (np.concatenate([X_train_branch_origin, X_train_branch_addition], axis=0),
               np.concatenate([X_train_trunk_origin, X_train_trunk_addition], axis=0))
    y_train = np.concatenate([y_train_origin, y_train_addition], axis=0)
    X_test = (np.tile(sensor_value, [101*101, 1]), 
              np.array([[a, b] for a in np.linspace(0, 1, 101) for b in np.linspace(0, 1, 101)]))
    y_test = data_test['y_test'][repeat]
    data = Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    net = dde.nn.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
        trainable_branch=True,
        trainable_trunk=[True, True, True],
    )
    net.apply_output_transform(dirichlet)
    net.build()
    net.inputs = [sensor_value, None]

    model = dde.Model(data, net)
    iterations = 3000
    model.compile("adam", lr=0.001, loss=loss_func, metrics=["l2 relative error"], 
                  decay=("inverse time", iterations // 5, 0.8))
    model.train(iterations=iterations, display_every=100, batch_size=10000, model_restore_path=f"model/model")

    return model.predict(X_test).reshape(1, 101*101)


if __name__ == "__main__":
    num_train = 100
    for ls_test in [0.2, 0.15, 0.1, 0.05]:
        data_train = np.load(f"../../../data/advection/adv_train_ls_0.5_101_101.npz")
        X_train_branch_origin = np.repeat(data_train["X_train0"], 101 * 101, axis=0)
        X_train_trunk_origin = np.tile(data_train["X_train1"], (1000, 1))
        y_train_origin = data_train["y_train"].reshape(-1, 1)
        output_list = []
        for repeat in range(100):
            output = apply(deeponet_together, (repeat, ls_test, num_train))
            output_list.append(output)
        output_list = np.concatenate(output_list, axis=0)
        np.save(f"predict_ft_obs_t_ls_{ls_test}.npy", output_list)
    