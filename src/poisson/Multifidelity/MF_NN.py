import os
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
import numpy as np
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


def mfnn(repeat, ls_test, num_train, weight_decay, y_lo_train):
    import deepxde as dde

    data_test = np.load(f"../../../data/poisson/poisson_test_ls_{ls_test}_num_{num_train}.npz")

    X_hi_train = data_test['x_trunk_test_select'][repeat]
    y_hi_train = data_test['y_test_select'][repeat]
    X_hi_test = data_test['x_trunk_test']
    y_hi_test = data_test['y_test'][repeat]
    X_lo_train = data_test['x_trunk_test']

    data = dde.data.MfDataSet(
        X_lo_train=X_lo_train,
        X_hi_train=X_hi_train,
        y_lo_train=y_lo_train,
        y_hi_train=y_hi_train,
        X_hi_test=X_hi_test,
        y_hi_test=y_hi_test,
    )

    net = dde.maps.MfNN(
        [2] + [256] * 3 + [1],
        [15] * 2 + [1],
        "relu",
        "Glorot uniform",
        regularization=["l2", weight_decay],
        residue=True,
        trainable_low_fidelity=True,
        trainable_high_fidelity=True,
    )

    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    losshistory, train_state = model.train(epochs=20000)

    dde.saveplot(losshistory, train_state, issave=False, isplot=False)
    return np.array(model.predict(X_hi_test))[1].reshape(1, 5082)


def main():
    ls_test = 0.2
    num_train = 50
    weight_decay = 1e-6
    d = np.loadtxt(f"../../../data/poisson/low_fidelity_{ls_test}.dat")
    output_list = []
    for repeat in range(100):
        y_lo_train = d[repeat][:, None]
        output = apply(mfnn, [repeat, ls_test, num_train, weight_decay, y_lo_train])
        output_list.append(output)
    output_list = np.concatenate(output_list, axis=0)
    np.save(f"predict_mfnn.npy", output_list)


if __name__ == "__main__":
    main()
