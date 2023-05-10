import numpy as np
import deepxde as dde


def solve_Burgers(XMAX, TMAX, NX, NT, NU, u0):
    """
   Returns the velocity field and distance for 1D non-linear Burgers equation, XMIN = 0, TMIN=0
   """

    DT = TMAX / (NT - 1)
    DX = XMAX / (NX - 1)

    # Initialise data structures
    u = np.zeros((NX - 1, NT))

    # Initial conditions
    u[:, 0] = u0[:-1]

    # Periodic boundary conditions
    I = np.eye(NX - 1)
    I1 = np.roll(I, 1, axis=0)
    I2 = np.roll(I, -1, axis=0)
    A = I2 - I1
    B = I1 + I2 - 2 * I

    # Numerical solution
    for n in range(0, NT - 1):
        u[:, n + 1] = u[:, n] - (DT / (2 * DX)) * np.dot(A, u[:, n]) * u[:, n] + NU * (DT / DX ** 2) * np.dot(B, u[:, n])
    u = np.concatenate([u, u[0:1, :]], axis=0)
    return u


def eval_s(sensor_values):
    return solve_Burgers(XMAX=1, TMAX=1, NX=201, NT=20001, NU=0.1, u0=sensor_values)


def gen_train(ls_train):
    m = 201
    T = 1
    num_train = 1000
    print("Generating operator data...", flush=True)
    space = dde.data.GRF(T=T, kernel="ExpSineSquared", length_scale=ls_train, N=2001, interp="cubic")
    features = space.random(num_train)
    sensors = np.linspace(0, 1, num=m)[:, None]
    sensor_values = space.eval_batch(features, sensors)
    X_branch_train = sensor_values.reshape(len(sensor_values), 201)[:, ::2]
    X_trunk_train = np.array([[a, b] for a in np.linspace(0, 1, 101) for b in np.linspace(0, 1, 101)])
    s = np.array(list(map(eval_s, sensor_values, )))
    y_train = s[:, ::2, ::200].reshape(len(sensor_values), 101 * 101)
    np.savez(f"burgers_train_ls_{ls_train}_101_101.npz", X_train0=X_branch_train, X_train1=X_trunk_train, y_train=y_train,)


def gen_test(ls_test):
    m = 201
    T = 1
    num_test = 100
    print("Generating operator data...", flush=True)
    space = dde.data.GRF(T=T, kernel="ExpSineSquared", length_scale=ls_test, N=2001, interp="cubic")
    features = space.random(num_test)
    sensors = np.linspace(0, 1, num=m)[:, None]
    sensor_values = space.eval_batch(features, sensors)
    X_branch_test = sensor_values.reshape(len(sensor_values), 201)[:, ::2]
    X_trunk_test = np.array([[a, b] for a in np.linspace(0, 1, 101) for b in np.linspace(0, 1, 101)])
    s = np.array(list(map(eval_s, sensor_values, )))
    y_test = s[:, ::2, ::200].reshape(len(sensor_values), 101 * 101)
    np.savez(f"burgers_test_ls_{ls_test}_101_101.npz", X_test0=X_branch_test, X_test1=X_trunk_test, y_test=y_test,)


def gen_observation(ls):
    indices = np.sort(np.random.randint(10201, size=(100, 100)))
    data_test = np.load(f"burgers_test_ls_{ls}_101_101.npz")
    x_branch_test = data_test["X_test0"]
    y_test = data_test["y_test"]
    x_trunk_test_select = []
    y_test_select = []
    xt = np.array([[a, b] for a in np.linspace(0, 1, 101) for b in np.linspace(0, 1, 101)])
    for i in range(100):
        index = indices[i]
        xt_selected = xt[index]
        x_trunk_test_select.append(xt_selected)
        y_select = y_test[i][index]
        y_test_select.append(y_select)
    x_trunk_test_select = np.concatenate(x_trunk_test_select, axis=0).reshape((100, 100, 2))
    y_test_select = np.concatenate(y_test_select, axis=0).reshape((100, 100, 1))
    np.savez(f"burgers_test_ls_{ls}_num_100.npz", x_branch_test=x_branch_test,
             x_trunk_test_select=x_trunk_test_select, y_test_select=y_test_select,
             y_test=y_test.reshape(100, 10201, 1))


if __name__ == "__main__":
    np.random.seed(0)
    gen_train(1.0)
    gen_test(0.6)
    gen_observation(0.6)
    gen_test(1.0)
