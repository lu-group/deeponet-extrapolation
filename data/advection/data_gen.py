import jax.numpy as jnp
from jax import random, vmap
from jax.config import config
from jax import lax
import numpy as np

m = 101

# Deinfe initial and boundary conditions for advection equation
# IC: f(x, 0)  = sin(pi x)
# BC: g(0, t) = sin (pi t / 2)
f = lambda x: jnp.sin(jnp.pi * x)
g = lambda t: jnp.sin(jnp.pi * t / 2)


# Advection solver
def solve_CVC(gp_sample):
    # Solve u_t + a(x) * u_x = 0
    # Wendroff for a(x)=V(x) - min(V(x)+ + 1.0, u(x,0)=f(x), u(0,t)=g(t)  (f(0)=g(0))
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    Nx = 501
    Nt = 501
    N = gp_sample.shape[0]
    X = jnp.linspace(xmin, xmax, N)[:, None]
    V = lambda x: jnp.interp(x, X.flatten(), gp_sample)

    # Create grid
    x = jnp.linspace(xmin, xmax, Nx)
    t = jnp.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    lam = dt / h

    # Compute advection velocity
    v_fn = lambda x: V(x) - V(x).min() + 1.0
    v = v_fn(x)

    # Initialize solution and apply initial & boundary conditions
    u = jnp.zeros([Nx, Nt])
    u = u.at[0, :].set(g(t))
    u = u.at[:, 0].set(f(x))

    # Compute finite difference operators
    a = (v[:-1] + v[1:]) / 2
    k = (1 - a * lam) / (1 + a * lam)
    K = jnp.eye(Nx - 1, k=0)
    K_temp = jnp.eye(Nx - 1, k=0)
    Trans = jnp.eye(Nx - 1, k=-1)

    def body_fn_x(i, carry):
        K, K_temp = carry
        K_temp = (-k[:, None]) * (Trans @ K_temp)
        K += K_temp
        return K, K_temp

    K, _ = lax.fori_loop(0, Nx - 2, body_fn_x, (K, K_temp))
    D = jnp.diag(k) + jnp.eye(Nx - 1, k=-1)

    def body_fn_t(i, u):
        b = jnp.zeros(Nx - 1)
        b = b.at[0].set(g(i * dt) - k[0] * g((i + 1) * dt))
        u = u.at[1:, i + 1].set(K @ (D @ u[1:, i] + b))
        return u

    UU = lax.fori_loop(0, Nt - 1, body_fn_t, u)
    UU = UU.T

    # Input sensor locations and measurements
    xx = jnp.linspace(xmin, xmax, m)
    u = v_fn(xx)

    return x, t, UU[::5, ::5], u


# Use double precision to generate data (due to GP sampling)
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = jnp.expand_dims(x1 / lengthscales, 1) - jnp.expand_dims(x2 / lengthscales, 0)
    r2 = jnp.sum(diffs ** 2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)


def generate_one_data(key, length_scale):
    xmin, xmax = 0.0, 1.0
    N = 501
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = jnp.linspace(xmin, xmax, N)[:, None]
    K = RBF(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter * jnp.eye(N))
    gp_sample = jnp.dot(L, random.normal(key, (N,)))

    x, t, UU, u = solve_CVC(gp_sample)

    u = u.reshape(-1)
    s = UU.T.reshape(-1)

    return u, s


def generate_data(key, Num, length_scale):
    config.update("jax_enable_x64", True)
    keys = random.split(key, Num)
    u, s = vmap(generate_one_data, (0, None))(keys, length_scale)
    y = jnp.array([[a, b] for a in jnp.linspace(0, 1, 101) for b in jnp.linspace(0, 1, 101)])
    config.update("jax_enable_x64", False)
    return u, y, s


def data_gen_train():
    key = random.PRNGKey(0)
    Num = 1000
    length_scale = 0.5
    X_branch, X_trunk, y = generate_data(key, Num, length_scale)
    print(X_branch.shape)
    print(X_trunk.shape)
    print(y.shape)
    jnp.savez("adv_train_ls_0.5_101_101.npz", X_train0=X_branch, X_train1=X_trunk, y_train=y)


def data_gen_test():
    for length_scale in [0.05, 0.1, 0.15, 0.2, 0.5]:
        key = random.PRNGKey(0)
        Num = 100
        X_branch, X_trunk, y = generate_data(key, Num, length_scale)
        print(X_branch.shape)
        print(X_trunk.shape)
        print(y.shape)
        jnp.savez(f"adv_test_ls_{length_scale}_101_101.npz", X_test0=X_branch, X_test1=X_trunk, y_test=y)


def gen_observation():
    np.random.seed(0)
    for ls in [0.05, 0.1, 0.15, 0.2]:
        indices = np.sort(np.random.randint(10201, size=(100, 100)))
        data_test = np.load(f"adv_test_ls_{ls}_101_101.npz")
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
        np.savez(f"adv_test_ls_{ls}_num_100.npz", x_branch_test=x_branch_test,
                 x_trunk_test_select=x_trunk_test_select, y_test_select=y_test_select,
                 y_test=y_test.reshape(100, 10201, 1))


def main():
    data_gen_train()
    data_gen_test()
    gen_observation()


if __name__ == "__main__":
    main()
