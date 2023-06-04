import numpy as np

num = 50
ls_test = 0.2
indices = np.sort(np.random.randint(5082, size=(100, num)))
data_test = np.load(f"poisson_test_ls_{ls_test}_5082.npz")
x_branch_test=data_test["X_test0"]
xt = data_test["X_test1"]
y_test = data_test["y_test"]
x_trunk_test_select = []
y_test_select = []

for i in range(100):
    index = indices[i]
    xt_selected = xt[index]
    x_trunk_test_select.append(xt_selected)
    y_select = y_test[i][index]
    y_test_select.append(y_select)
x_trunk_test_select = np.concatenate(x_trunk_test_select, axis=0).reshape(100, num, 2)
y_test_select = np.concatenate(y_test_select, axis=0).reshape(100, num, 1)
np.savez(f"poisson_test_ls_{ls_test}_num_{num}.npz", x_branch_test=x_branch_test,
         x_trunk_test_select=x_trunk_test_select, x_trunk_test=xt,
         y_test_select=y_test_select, y_test=y_test.reshape(100, 5082, 1))
