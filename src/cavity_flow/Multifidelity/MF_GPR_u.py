import os
os.environ['DDE_BACKEND'] = 'tensorflow.compat.v1'
import numpy as np
import GPy
import emukit.multi_fidelity
import emukit.test_functions
from emukit.model_wrappers.gpy_model_wrappers import GPyMultiOutputWrapper
from emukit.multi_fidelity.convert_lists_to_array import convert_x_list_to_array
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
import deepxde as dde


class LinearMFGP(object):
    def __init__(self, noise=None, n_optimization_restarts=10):
        self.noise = noise
        self.n_optimization_restarts = n_optimization_restarts
        self.model = None

    def train(self, x_l, y_l, x_h, y_h):
        # Construct a linear multi-fidelity model
        X_train, Y_train = convert_xy_lists_to_arrays([x_l, x_h], [y_l, y_h])
        kernels = [GPy.kern.src.sde_matern.sde_Matern32(x_l.shape[1]), GPy.kern.src.sde_matern.sde_Matern32(x_h.shape[1])]
        kernel = emukit.multi_fidelity.kernels.LinearMultiFidelityKernel(kernels)
        gpy_model = GPyLinearMultiFidelityModel(
            X_train, Y_train, kernel, n_fidelities=2
        )
        if self.noise is not None:
            gpy_model.mixed_noise.Gaussian_noise.fix(self.noise)
            gpy_model.mixed_noise.Gaussian_noise_1.fix(self.noise)

        # Wrap the model using the given 'GPyMultiOutputWrapper'
        self.model = GPyMultiOutputWrapper(
            gpy_model, 2, n_optimization_restarts=self.n_optimization_restarts
        )
        # Fit the model
        self.model.optimize()

    def predict(self, x):
        # Convert x_plot to its ndarray representation
        X = convert_x_list_to_array([x, x])
        X_l = X[: len(x)]
        X_h = X[len(x) :]

        # Compute mean predictions and associated variance
        lf_mean, lf_var = self.model.predict(X_l)
        lf_std = np.sqrt(lf_var)
        hf_mean, hf_var = self.model.predict(X_h)
        hf_std = np.sqrt(hf_var)
        return lf_mean, lf_std, hf_mean, hf_std


def mfgpr(repeat, num_train, x_train_l, y_train_l):
    data_test = np.load(f"../../../data/cavity_flow/cavity_extrapolation_num_{num_train}_10times.npz")

    x_train_h = data_test['x_trunk_test_select'][repeat]
    y_train_h = data_test['u_test_select'][repeat]
    x_test_h = data_test['x_trunk_test'][repeat]
    y_test_h = data_test['u_test'][repeat]

    model = LinearMFGP(noise=0, n_optimization_restarts=5)
    model.train(x_train_l, y_train_l, x_train_h, y_train_h)
    lf_mean, lf_std, hf_mean, hf_std = model.predict(x_test_h)
    l2 = dde.metrics.l2_relative_error(y_test_h, hf_mean)
    print(f"l2: {l2}")
    return hf_mean.reshape(1, 101*101)


def main():
    num_train = 100
    xy = np.load(f"../../../data/cavity_flow/cavity_extrapolation_num_{num_train}_10times.npz")["x_trunk_test"]
    d = np.loadtxt(f"../../../data/cavity_flow/low_fidelity_u_10times.dat")
    output_list = []
    for repeat in range(100):
        x_train_l = xy[repeat].reshape(101, 101, 2)[::5, ::5, :].reshape(-1, 2)
        y_train_l = d[repeat].reshape(101, 101)[::5, ::5].reshape(-1, 1)
        output = dde.utils.apply(mfgpr, [repeat, num_train, x_train_l, y_train_l])
        output_list.append(output)
    output_list = np.concatenate(output_list, axis=0)
    np.save(f"predict_mfgpr_u.npy", output_list)


if __name__ == "__main__":
    main()
