import numpy as np


num = 100
branch=[]
for k in np.linspace(0.41, 0.50, 10):
    branch.append(k*(1-np.linspace(0,1,101).reshape(1,101)))
x_branch_test=np.repeat(np.concatenate(branch), 10, axis=0)

indices = np.sort(np.random.randint(10201, size=(100, num)))
data_test = np.load(f"cavity_extrapolation.npz")
xy = np.repeat(data_test["trunk_extrapolation"].reshape(10, 101*101, 2), 10, axis=0)
u_test = np.repeat(data_test["u_extrapolation"].reshape(10, 101*101, 1), 10, axis=0)
v_test = np.repeat(data_test["v_extrapolation"].reshape(10, 101*101, 1), 10, axis=0)
x_trunk_test_select = []
u_test_select = []
v_test_select = []
for i in range(100):
    index = indices[i]
    xt_selected = xy[i][index]
    x_trunk_test_select.append(xt_selected)

    u_select = u_test[i][index]
    u_test_select.append(u_select)

    v_select = v_test[i][index]
    v_test_select.append(v_select)
x_trunk_test_select = np.concatenate(x_trunk_test_select, axis=0).reshape(100, num, 2)
u_test_select = np.concatenate(u_test_select, axis=0).reshape(100, num, 1)
v_test_select = np.concatenate(v_test_select, axis=0).reshape(100, num, 1)
np.savez(f"cavity_extrapolation_num_{num}_10times.npz", x_branch_test=x_branch_test, x_trunk_test=xy,
         x_trunk_test_select=x_trunk_test_select, u_test_select=u_test_select,
         v_test_select=v_test_select, u_test=u_test.reshape(100, 10201, 1), v_test=v_test.reshape(100, 10201, 1))
