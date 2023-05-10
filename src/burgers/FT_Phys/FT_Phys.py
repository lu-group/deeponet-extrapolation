import os
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
import numpy as np
from scipy.interpolate import griddata
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


def ft_phys(lr, boolean_value, sensor_value, xt, output_func):
    import deepxde as dde

    def periodic(x):
        return tf.concat([tf.math.cos(x[:, 0:1] * 2 * np.pi), tf.math.sin(x[:, 0:1] * 2 * np.pi),
                          tf.math.cos(2 * x[:, 0:1] * 2 * np.pi), tf.math.sin(2 * x[:, 0:1] * 2 * np.pi),
                          tf.math.cos(3 * x[:, 0:1] * 2 * np.pi), tf.math.sin(3 * x[:, 0:1] * 2 * np.pi),
                          tf.math.cos(4 * x[:, 0:1] * 2 * np.pi), tf.math.sin(4 * x[:, 0:1] * 2 * np.pi), x[:, 1:2]], 1)

    def gelu(x):
        return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.1 * dy_xx

    def func(x_input):
        uu = output_func.reshape(-1, 1)
        return griddata(xt, uu, x_input, method="cubic")

    x0 = np.array([[a, 0] for a in np.linspace(0, 1, 101)])
    y0 = sensor_value.reshape(101, 1)
    ic = dde.icbc.PointSetBC(x0, y0)

    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    data = dde.data.TimePDE(geomtime, pde, [ic], num_domain=2500, solution=func, num_test=10000)

    net = dde.maps.DeepONet(
        [101, 100, 100, 100],
        [2, 100, 100, 100],
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
        trainable_branch=boolean_value[0],
        trainable_trunk=[boolean_value[1], boolean_value[2], boolean_value[3]],
    )

    net.apply_feature_transform(periodic)
    net.build()
    net.inputs = [sensor_value, None]

    model = dde.Model(data, net)
    model.compile("adam", lr=lr, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=5000, display_every=100,
                                           model_restore_path="model/model")

    dde.saveplot(losshistory, train_state, issave=False, isplot=False)

    metrics_test = np.array(losshistory.metrics_test).reshape(-1, 1)
    best_metrics = train_state.best_metrics[0]
    return metrics_test, best_metrics


def pure_pinn(lr, sensor_value, xt, output_func):
    import deepxde as dde

    def periodic(x):
        return tf.concat([tf.math.cos(x[:, 0:1] * 2 * np.pi), tf.math.sin(x[:, 0:1] * 2 * np.pi),
                          tf.math.cos(2 * x[:, 0:1] * 2 * np.pi), tf.math.sin(2 * x[:, 0:1] * 2 * np.pi),
                          tf.math.cos(3 * x[:, 0:1] * 2 * np.pi), tf.math.sin(3 * x[:, 0:1] * 2 * np.pi),
                          tf.math.cos(4 * x[:, 0:1] * 2 * np.pi), tf.math.sin(4 * x[:, 0:1] * 2 * np.pi), x[:, 1:2]], 1)

    def gelu(x):
        return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.1 * dy_xx

    def func(x_input):
        uu = output_func.reshape(-1, 1)
        return griddata(xt, uu, x_input, method="cubic")

    x0 = np.array([[a, 0] for a in np.linspace(0, 1, 101)])
    y0 = sensor_value.reshape(101, 1)
    ic = dde.icbc.PointSetBC(x0, y0)
    geom = dde.geometry.Interval(0, 1)
    timedomain = dde.geometry.TimeDomain(0, 1)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)
    data = dde.data.TimePDE(geomtime, pde, [ic], num_domain=2500, solution=func, num_test=10000)

    layer_size = [2] + [100] * 3 + [1]
    activation = gelu
    initializer = "Glorot uniform"
    net = dde.maps.FNN(layer_size, activation, initializer)

    net.apply_feature_transform(periodic)
    model = dde.Model(data, net)
    model.compile("adam", lr=lr, metrics=["l2 relative error"])

    losshistory, train_state = model.train(epochs=5000, display_every=100)

    dde.saveplot(losshistory, train_state, issave=False, isplot=False)
    metrics_test = np.array(losshistory.metrics_test).reshape(-1, 1)
    best_metrics = train_state.best_metrics[0]
    output = metrics_test
    return output, best_metrics


def main():
    data_test = np.load(f"../../../data/burgers/burgers_test_ls_0.6_101_101.npz")
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
