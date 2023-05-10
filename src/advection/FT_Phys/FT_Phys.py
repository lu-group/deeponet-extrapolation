import os
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
import numpy as np
from scipy.interpolate import griddata
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
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


def ft_phys(lr, boolean_value, sensor_value, xt, output_func):
    import deepxde as dde

    def dirichlet(inputs, outputs):
        x_trunk = inputs[1]
        x, t = x_trunk[:, 0:1], x_trunk[:, 1:2]
        return 4 * x * t * outputs + tf.sin(np.pi * x) + tf.sin(0.5 * np.pi * t)

    def gelu(x):
        return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def pde(x, y):
        dy = tf.gradients(y, x)[0]
        dy_x, dy_t = dy[:, 0:1], dy[:, 1:]
        u = tfp.math.batch_interp_regular_1d_grid(x[:, :1], 0, 1, np.float32(sensor_value))
        return dy_t + u * dy_x

    def func(x_input):
        uu = output_func.reshape(-1, 1)
        return griddata(xt, uu, x_input, method="cubic")

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    data = dde.data.TimePDE(geomtime, pde, [], num_domain=2500, solution=func, num_test=10000)

    net = dde.maps.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
        trainable_branch=boolean_value[0],
        trainable_trunk=[boolean_value[1], boolean_value[2], boolean_value[3]],
    )

    net.apply_output_transform(dirichlet)
    net.build()
    net.inputs = [sensor_value, None]

    model = dde.Model(data, net)
    model.compile("adam", lr=lr, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=5000, display_every=50,
                                           model_restore_path="model/model")

    dde.saveplot(losshistory, train_state, issave=False, isplot=False)

    metrics_test = np.array(losshistory.metrics_test).reshape(-1, 1)
    best_metrics = train_state.best_metrics[0]
    return metrics_test, best_metrics


def pure_pinn(lr, sensor_value, xt, output_func):
    import deepxde as dde

    def dirichlet(inputs, outputs):
        x, t = inputs[:, 0:1], inputs[:, 1:2]
        return 4 * x * t * outputs + tf.sin(np.pi * x) + tf.sin(0.5 * np.pi * t)

    def gelu(x):
        return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def pde(x, y):
        dy = tf.gradients(y, x)[0]
        dy_x, dy_t = dy[:, 0:1], dy[:, 1:]
        u = tfp.math.batch_interp_regular_1d_grid(x[:, :1], 0, 1, np.float32(sensor_value))
        return dy_t + u * dy_x

    def func(x_input):
        uu = output_func.reshape(-1, 1)
        return griddata(xt, uu, x_input, method="cubic")

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    data = dde.data.TimePDE(geomtime, pde, [], num_domain=2500, solution=func, num_test=10000)

    layer_size = [2] + [100] * 3 + [1]
    activation = gelu
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    net.apply_output_transform(dirichlet)
    model = dde.Model(data, net)
    model.compile("adam", lr=lr, metrics=["l2 relative error"])

    losshistory, train_state = model.train(epochs=5000, display_every=50)

    dde.saveplot(losshistory, train_state, issave=False, isplot=False)
    metrics_test = np.array(losshistory.metrics_test).reshape(-1, 1)
    best_metrics = train_state.best_metrics[0]
    output = metrics_test
    return output, best_metrics


def main():
    for ls_test in [0.2, 0.15, 0.1, 0.05]:
        data_test = np.load(f"../../../data/advection/adv_test_ls_{ls_test}_101_101.npz")
        sensor_values = data_test['X_test0']
        xt = data_test['X_test1']
        y_test = data_test['y_test']
        lr = 0.001
        for i in range(len(sensor_values)):
            sensor_value = sensor_values[i]
            output_func = y_test[i]
            for boolean_value in [[True, True, True, True], [True, False, False, False], [False, True, True, True],
                                  [False, False, False, True], [False, False, True, True]]:
                apply(ft_phys, (lr, boolean_value, sensor_value, xt, output_func))
            apply(pure_pinn, (lr, sensor_value, xt, output_func))


if __name__ == "__main__":
    main()
