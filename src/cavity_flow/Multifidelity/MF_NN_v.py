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


def mfnn(repeat, num_train, weight_decay, X_lo_train, y_lo_train):
    import deepxde as dde

    data_test = np.load(f"../../../data/cavity_flow/cavity_extrapolation_num_{num_train}_10times.npz")

    X_hi_train = data_test['x_trunk_test_select'][repeat]
    y_hi_train = data_test['v_test_select'][repeat]
    X_hi_test = data_test['x_trunk_test'][repeat]
    y_hi_test = data_test['v_test'][repeat]

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
    return np.array(model.predict(X_hi_test))[1].reshape(1, 101*101)


def main():
    num_train = 100
    xy = np.load(f"../../../data/cavity_flow/cavity_extrapolation_num_{num_train}_10times.npz")["x_trunk_test"]
    weight_decay = 1e-6
    d = np.loadtxt(f"../../../data/cavity_flow/low_fidelity_v_10times.dat")
    output_list = []
    for repeat in range(100):
        X_lo_train = xy[repeat].reshape(-1, 2)
        y_lo_train = d[repeat][:, None]
        output = apply(mfnn, [repeat, num_train, weight_decay, X_lo_train, y_lo_train])
        output_list.append(output)
    output_list = np.concatenate(output_list, axis=0)
    np.save(f"predict_mfnn_v.npy", output_list)


if __name__ == "__main__":
    main()
