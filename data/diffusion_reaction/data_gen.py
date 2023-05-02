import numpy as np
from deepxde.data import GRF


def solve_ADR(xmin, xmax, tmin, tmax, k, v, g, dg, f, u0, Nx, Nt):
    """Solve 1D
    u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x, t)
    with zero boundary condition.
    """

    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    k = k(x)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v = v(x)
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond
    f = f(x[:, None], t)

    u = np.zeros((Nx, Nt))
    u[:, 0] = u0(x)
    for i in range(Nt - 1):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1, i] + 0.5 * f[1:-1, i + 1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u[1:-1, i + 1] = np.linalg.solve(A, b1 + b2)
    return x, t, u


def eval_s(sensor_value):
    """Compute s(x, t) over m * Nt points for a `sensor_value` of `u`.
    """
    return solve_ADR(
        0,
        1,
        0,
        T,
        lambda x: D * np.ones_like(x),
        lambda x: np.zeros_like(x),
        lambda u: k * u ** 2,
        lambda u: 2 * k * u,
        lambda x, t: np.tile(sensor_value[:, None], (1, len(t))),
        lambda x: np.zeros_like(x),
        len(sensor_value),
        Nt,
    )[2]


def gen_train(ls):
    ls = round(ls, 1)
    print("Generating operator data...", flush=True)
    space = GRF(T, length_scale=ls, N=1000 * T, interp="cubic")
    features = space.random(num_train)
    sensors = np.linspace(0, 1, num=m)[:, None]
    sensor_values = space.eval_batch(features, sensors)
    s_values = list(map(eval_s, sensor_values))
    for j in range(len(s_values)):
        s_values[j] = np.concatenate(s_values[j])
    s_values = np.array(s_values)
    x = np.linspace(0, 1, m)
    t = np.linspace(0, T, Nt)
    xt = np.array([[a, b] for a in x for b in t])
    np.savez(f"dr_train_ls_{ls}_101_101.npz", X_train0=sensor_values, X_train1=xt, y_train=s_values)


def gen_test(ls):
    ls = round(ls, 1)
    print("Generating operator data...", flush=True)
    space = GRF(T, length_scale=ls, N=1000 * T, interp="cubic")
    features = space.random(num_test)
    sensors = np.linspace(0, 1, num=m)[:, None]
    sensor_values = space.eval_batch(features, sensors)
    s_values = list(map(eval_s, sensor_values))
    for j in range(len(s_values)):
        s_values[j] = np.concatenate(s_values[j])
    s_values = np.array(s_values)
    x = np.linspace(0, 1, m)
    t = np.linspace(0, T, Nt)
    xt = np.array([[a, b] for a in x for b in t])
    np.savez(f"dr_test_ls_{ls}_101_101.npz", X_test0=sensor_values, X_test1=xt, y_test=s_values)


def gen_observation(ls):
    ls = round(ls, 1)
    indices = np.sort(np.random.randint(10201, size=(100, 100)))
    data_test = np.load(f"dr_test_ls_{ls}_101_101.npz")
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
    np.savez(f"dr_test_ls_{ls}_num_100.npz", x_branch_test=x_branch_test,
             x_trunk_test_select=x_trunk_test_select, y_test_select=y_test_select,
             y_test=y_test.reshape(100, 10201, 1))


if __name__ == "__main__":
    T = 1
    D = 0.01
    k = 0.01
    Nt = 101
    num_train = 1000
    num_test = 100
    m = 101

    np.random.seed(0)
    gen_train(0.5)
    gen_test(0.2)
    gen_observation(0.2)
    gen_test(0.5)
