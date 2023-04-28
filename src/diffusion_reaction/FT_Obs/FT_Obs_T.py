import os
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
import numpy as np
from multiprocessing import Pool


def apply(func, args=None, kwds=None):
    """Launch a new process to call the function.

    This can be used to clear Tensorflow GPU memory after model execution:
    https://stackoverflow.com/questions/39758094/clearing-tensorflow-gpu-memory-after-model-execution
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


def deeponet_together(repeat, ls_test, num_train, k_coefficient):
    import deepxde as dde
    from deepxde.backend import tf
    def gelu(x):
        return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

    def dirichlet(inputs, outputs):
        x_trunk = inputs[1]
        x, t = x_trunk[:, 0:1], x_trunk[:, 1:2]
        return 10 * x * (1 - x) * t * (outputs + 1)

    ls = 0.5
    k = round(k_coefficient * 1E5 / num_train)
    data_test = np.load(f"../../../data/diffusion_reaction/dr_test_ls_{ls_test}_num_{num_train}.npz")
    sensor_value = data_test['x_branch_test'][repeat]
    X_branch_train_addition = np.tile(sensor_value, [k * num_train, 1])
    X_trunk_train_addition = np.tile(data_test['x_trunk_test_select'][repeat], [k, 1])
    y_train_addition = np.tile(data_test['y_test_select'][repeat], [k, 1])
    data_train = np.load(f"../../../data/diffusion_reaction/dr_train_ls_{ls}_1e5.npz")
    X_train = (np.concatenate([data_train["X_train0"], X_branch_train_addition], axis=0),
               np.concatenate([data_train["X_train1"], X_trunk_train_addition], axis=0))
    y_train = np.concatenate([data_train["y_train"], y_train_addition], axis=0)
    X_test = (np.tile(sensor_value, [101*101, 1]),
              np.array([[a, b] for a in np.linspace(0, 1, 101) for b in np.linspace(0, 1, 101)]))
    y_test = data_test['y_test'][repeat]

    data = dde.data.Triple(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    net = dde.maps.DeepONet(
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
    epochs = 3000
    model.compile("adam", lr=0.001, metrics=["l2 relative error"], decay=("inverse time", epochs // 5, 0.8))

    model.train(epochs=epochs, display_every=100, batch_size=10000, model_restore_path=f"model/model")

    return model.predict(X_test).reshape(1, 101*101)


def main():
    ls_test = 0.2
    num_train = 100
    k_coefficient = 0.3
    output_list = []
    for repeat in range(100):
        output = apply(deeponet_together, (repeat, ls_test, num_train, k_coefficient))
        output_list.append(output)
    output_list = np.concatenate(output_list, axis=0)
    np.save(f"predict_ft_obs_t.npy", output_list)


if __name__ == "__main__":
    main()
