import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import time
from tqdm import tqdm

import calculateloss as cl
from basic_model import DeepModel_single


# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.net = torch.nn.Sequential(
#             torch.nn.Linear(2, 40),
#             torch.nn.Linear(40, 40),
#             torch.nn.Tanh(),
#             torch.nn.Linear(40, 40),
#             torch.nn.Tanh(),
#             torch.nn.Linear(40, 40),
#             torch.nn.Tanh(),
#             torch.nn.Linear(40, 40),
#             torch.nn.Tanh(),
#             torch.nn.Linear(40, 40),
#             torch.nn.Tanh(),
#             torch.nn.Linear(40, 40),
#             torch.nn.Tanh(),
#             torch.nn.Linear(40, 9),
#         )

#     def forward(self, x):
#         return self.net(x)


class Net(DeepModel_single):
    def __init__(self, planes):
        super(Net, self).__init__(planes, active=nn.Tanh())


def read_data():
    data_internal = pd.read_csv("./data/data_internal.csv", header=None)
    data_internal = data_internal.to_numpy()
    data_left = pd.read_csv("./data/data_left.csv", header=None)
    data_left = data_left.to_numpy()
    data_right = pd.read_csv("./data/data_right.csv", header=None)
    data_right = data_right.to_numpy()
    data_up = pd.read_csv("./data/data_up.csv", header=None)
    data_up = data_up.to_numpy()
    data_down = pd.read_csv("./data/data_down.csv", header=None)
    data_down = data_down.to_numpy()

    return data_internal, data_left, data_right, data_up, data_down


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    setup_seed(1234)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    R = 8.314
    T = 200
    RT = R * T
    tau = 1.402485925e-5

    N_internal = 2000
    N_BC_horizontal = 100
    N_BC_vertical = 200
    Model_eq = Net(planes=[2] + 6 * [40] + [9]).to(device)
    Model_neq = Net(planes=[2] + 6 * [40] + [9]).to(device)

    data = read_data()
    data = list(map(np.random.permutation, data))  # 打乱数据
    internal = data[0]
    left = data[1]
    right = data[2]
    up = data[3]
    down = data[4]

    internal = internal[:N_internal]
    left = left[:N_BC_horizontal]
    right = right[:N_BC_horizontal]
    up = up[:N_BC_vertical]
    down = down[:N_BC_vertical]

    x_internal = (
        torch.tensor(internal[:, 0], dtype=torch.float32).to(device).reshape(-1, 1)
    )
    y_internal = (
        torch.tensor(internal[:, 1], dtype=torch.float32).to(device).reshape(-1, 1)
    )
    x_left = torch.tensor(left[:, 0], dtype=torch.float32).to(device).reshape(-1, 1)
    y_left = torch.tensor(left[:, 1], dtype=torch.float32).to(device).reshape(-1, 1)
    x_right = torch.tensor(right[:, 0], dtype=torch.float32).to(device).reshape(-1, 1)
    y_right = torch.tensor(right[:, 1], dtype=torch.float32).to(device).reshape(-1, 1)
    x_up = torch.tensor(up[:, 0], dtype=torch.float32).to(device).reshape(-1, 1)
    y_up = torch.tensor(up[:, 1], dtype=torch.float32).to(device).reshape(-1, 1)
    x_down = torch.tensor(down[:, 0], dtype=torch.float32).to(device).reshape(-1, 1)
    y_down = torch.tensor(down[:, 1], dtype=torch.float32).to(device).reshape(-1, 1)

    rho_internal = (
        torch.tensor(internal[:, 4], dtype=torch.float32).to(device).reshape(-1, 1)
    )
    u_internal = (
        torch.tensor(internal[:, 5], dtype=torch.float32).to(device).reshape(-1, 1)
    )
    v_internal = (
        torch.tensor(internal[:, 6], dtype=torch.float32).to(device).reshape(-1, 1)
    )
    temp_internal = (
        torch.tensor(internal[:, 7], dtype=torch.float32).to(device).reshape(-1, 1)
    )
    press_internal = (
        torch.tensor(internal[:, 8], dtype=torch.float32).to(device).reshape(-1, 1)
    )

    rho_left = torch.tensor(left[:, 4], dtype=torch.float32).to(device).reshape(-1, 1)
    u_left = torch.tensor(left[:, 5], dtype=torch.float32).to(device).reshape(-1, 1)
    v_left = torch.tensor(left[:, 6], dtype=torch.float32).to(device).reshape(-1, 1)
    temp_left = torch.tensor(left[:, 7], dtype=torch.float32).to(device).reshape(-1, 1)
    press_left = torch.tensor(left[:, 8], dtype=torch.float32).to(device).reshape(-1, 1)

    rho_right = torch.tensor(right[:, 4], dtype=torch.float32).to(device).reshape(-1, 1)
    u_right = torch.tensor(right[:, 5], dtype=torch.float32).to(device).reshape(-1, 1)
    v_right = torch.tensor(right[:, 6], dtype=torch.float32).to(device).reshape(-1, 1)
    temp_right = (
        torch.tensor(right[:, 7], dtype=torch.float32).to(device).reshape(-1, 1)
    )
    press_right = (
        torch.tensor(right[:, 8], dtype=torch.float32).to(device).reshape(-1, 1)
    )

    rho_up = torch.tensor(up[:, 4], dtype=torch.float32).to(device).reshape(-1, 1)
    u_up = torch.tensor(up[:, 5], dtype=torch.float32).to(device).reshape(-1, 1)
    v_up = torch.tensor(up[:, 6], dtype=torch.float32).to(device).reshape(-1, 1)
    temp_up = torch.tensor(up[:, 7], dtype=torch.float32).to(device).reshape(-1, 1)
    press_up = torch.tensor(up[:, 8], dtype=torch.float32).to(device).reshape(-1, 1)

    rho_down = torch.tensor(down[:, 4], dtype=torch.float32).to(device).reshape(-1, 1)
    u_down = torch.tensor(down[:, 5], dtype=torch.float32).to(device).reshape(-1, 1)
    v_down = torch.tensor(down[:, 6], dtype=torch.float32).to(device).reshape(-1, 1)
    temp_down = torch.tensor(down[:, 7], dtype=torch.float32).to(device).reshape(-1, 1)
    press_down = torch.tensor(down[:, 8], dtype=torch.float32).to(device).reshape(-1, 1)

    Opt_eq = torch.optim.Adam(Model_eq.parameters(), lr=0.001)
    Opt_neq = torch.optim.Adam(Model_neq.parameters(), lr=0.001)

    start_time = time.time()

    _tqdm = tqdm(range(100000))
    for i in _tqdm:
        Opt_eq.zero_grad()
        Opt_neq.zero_grad()
        Loss = (
            cl.loss_PDEs(Model_eq, Model_neq, x_internal, y_internal, device)
            + cl.loss_BC_left(
                Model_eq,
                Model_neq,
                x_left,
                y_left,
                rho_left,
                u_left,
                v_left,
                temp_left,
                press_left,
            )
            + cl.loss_BC_right(
                Model_eq,
                Model_neq,
                x_right,
                y_right,
                rho_right,
                u_left,
                v_right,
                temp_right,
                press_right,
            )
            + cl.loss_BC_up(
                Model_eq, Model_neq, x_up, y_up, rho_up, u_left, v_up, temp_up, press_up
            )
            + cl.loss_BC_down(
                Model_eq,
                Model_neq,
                x_down,
                y_down,
                rho_down,
                u_left,
                v_down,
                temp_down,
                press_down,
            )
        )
        ## Loss = ()
        Loss.backward()
        Opt_eq.step()
        Opt_neq.step()
        _tqdm.set_postfix(loss="{:.6f}".format(Loss.item()))

    print("Time:", time.time() - start_time)
    print("Final loss:", Loss.item())

    torch.save(Model_eq, "./resualt/Model_eq_2.pth")
    torch.save(Model_neq, "./resualt/Model_neq_2.pth")
