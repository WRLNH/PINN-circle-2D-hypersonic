import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time
import random
from tqdm import tqdm

from basic_model import DeepModel_single, gradients
from D2Q9 import D2Q9_Model


class Net(DeepModel_single):
    def __init__(self, planes):
        super(Net, self).__init__(planes, active=nn.Tanh())


def CalResidualsLoss(in_var, out_eq, out_neq):
    loss = nn.MSELoss()

    f = out_eq + out_neq / 100000
    f_eq = out_eq
    f_neq = out_neq / 100000
    R = torch.zeros_like(f)
    dfda = gradients(f, in_var)
    dfdx, dfdy = dfda[..., 0, :], dfda[..., 1, :]
    for i in range(0, 9):
        R[:, i] = (
            D2Q9_Model.xi[i][0] * dfdx[:, i]
            + D2Q9_Model.xi[i][1] * dfdy[:, i]
            + 1 / tau * f_neq[:, i]
        )
    R = torch.sum(R, dim=1)
    cond = torch.zeros_like(R)
    return loss(R, cond)


def CalBCLoss(in_var, out_eq, out_neq, fields):
    loss = nn.MSELoss()

    out_eq_BC = out_eq[
        N_internal : N_internal + 2 * N_BC_horizontal + 2 * N_BC_vertical
    ]
    out_neq_BC = (
        out_neq[N_internal : N_internal + 2 * N_BC_horizontal + 2 * N_BC_vertical]
        / 100000
    )
    out_BC = out_eq_BC + out_neq_BC
    BC_exact = fields[N_internal : N_internal + 2 * N_BC_horizontal + 2 * N_BC_vertical]
    in_BC = in_var[N_internal : N_internal + 2 * N_BC_horizontal + 2 * N_BC_vertical]

    """Calculate L_mBC"""
    rho_pred = torch.sum(out_eq_BC, dim=1)
    rho_exact = BC_exact[:, 2]
    u_exact = BC_exact[:, 3]
    v_exact = BC_exact[:, 4]
    loss_rho = loss(rho_pred, rho_exact)

    rhou = torch.zeros_like(out_eq_BC)
    rhov = torch.zeros_like(out_eq_BC)
    for i in range(0, 9):
        rhou[:, i] = D2Q9_Model.xi[i][0] * out_eq_BC[:, i]
        rhov[:, i] = D2Q9_Model.xi[i][1] * out_eq_BC[:, i]
    rhou_pred = torch.sum(rhou, dim=1)
    rhov_pred = torch.sum(rhov, dim=1)
    loss_rhou = loss(rhou_pred, rho_exact * u_exact)
    loss_rhov = loss(rhov_pred, rho_exact * v_exact)

    # """Calculate L_fBC"""
    # in_left = in_BC[:N_BC_horizontal]
    # in_right = in_BC[N_BC_horizontal : 2 * N_BC_horizontal]
    # in_up = in_BC[2 * N_BC_horizontal : 2 * N_BC_horizontal + N_BC_vertical]
    # in_down = in_BC[2 * N_BC_horizontal + N_BC_vertical :]
    # out_left = out_eq_BC[:N_BC_horizontal]
    # out_right = out_eq_BC[N_BC_horizontal : 2 * N_BC_horizontal]
    # out_up = out_eq_BC[2 * N_BC_horizontal : 2 * N_BC_horizontal + N_BC_vertical]
    # out_down = out_eq_BC[2 * N_BC_horizontal + N_BC_vertical :]

    # # 对于左边界只计算i = [3, 6, 7]的点
    # f_eq = torch.zeros(out_left.shape[0], 3)
    # f_neq = torch.zeros(out_left.shape[0], 3)
    # temp = 0
    # range = [3, 6, 7]
    # for i in range:
    #     f_eq[:, temp] = (
    #         D2Q9_Model.w[i]
    #         * rho_exact[:N_BC_horizontal]
    #         * (
    #             1
    #             + (
    #                 D2Q9_Model.xi[i][0] * u_exact[0:N_BC_horizontal]
    #                 + D2Q9_Model.xi[i][1] * v_exact[0:N_BC_horizontal]
    #             )
    #             / RT
    #             + (
    #                 D2Q9_Model.xi[i][0] * u_exact[0:N_BC_horizontal]
    #                 + D2Q9_Model.xi[i][1] * v_exact[0:N_BC_horizontal]
    #             )
    #             ** 2
    #             / (2 * RT**2)
    #             - (u_exact[0:N_BC_horizontal] ** 2 + v_exact[0:N_BC_horizontal] ** 2)
    #             / (2 * RT)
    #         )
    #     )
    #     temp += 1
    # f_neq = (
    #     -tau
    #     * (
    #         D2Q9_Model.xi[i][0] * gradients(f_eq, in_left[:, 0])
    #         + D2Q9_Model.xi[i][1] * gradients(f_eq, in_left[:, 1])
    #     )
    #     * 100000
    # )  # 将F_neq放大至O(1)量级
    # col1 = out_neq_BC[:N_BC_horizontal][:, 3].unsqueeze(1)
    # col2 = out_neq_BC[:N_BC_horizontal][:, 6].unsqueeze(1)
    # col3 = out_neq_BC[:N_BC_horizontal][:, 7].unsqueeze(1)
    # f_neq_left = torch.cat((col1, col2, col3), dim=1)
    # loss_left = loss(f_neq_left, f_neq)

    # # 对于右边界只计算i = [1, 5, 8]的点
    # f_eq = torch.zeros(out_right.shape[0], 3)
    # f_neq = torch.zeros(out_right.shape[0], 3)
    # temp = 0
    # range = [1, 5, 8]
    # for i in range:
    #     f_eq[:, temp] = (
    #         D2Q9_Model.w[i]
    #         * rho_exact[N_BC_horizontal : 2 * N_BC_horizontal]
    #         * (
    #             1
    #             + (
    #                 D2Q9_Model.xi[i][0] * u_exact[N_BC_horizontal : 2 * N_BC_horizontal]
    #                 + D2Q9_Model.xi[i][1]
    #                 * v_exact[N_BC_horizontal : 2 * N_BC_horizontal]
    #             )
    #             / RT
    #             + (
    #                 D2Q9_Model.xi[i][0] * u_exact[N_BC_horizontal : 2 * N_BC_horizontal]
    #                 + D2Q9_Model.xi[i][1]
    #                 * v_exact[N_BC_horizontal : 2 * N_BC_horizontal]
    #             )
    #             ** 2
    #             / (2 * RT**2)
    #             - (
    #                 u_exact[N_BC_horizontal : 2 * N_BC_horizontal] ** 2
    #                 + v_exact[N_BC_horizontal : 2 * N_BC_horizontal] ** 2
    #             )
    #             / (2 * RT)
    #         )
    #     )
    #     temp += 1
    # f_neq = (
    #     -tau
    #     * (
    #         D2Q9_Model.xi[i][0] * gradients(f_eq, in_right[:, 0])
    #         + D2Q9_Model.xi[i][1] * gradients(f_eq, in_right[:, 1])
    #     )
    #     * 100000
    # )  # 将F_neq放大至O(1)量级
    # col1 = out_neq_BC[N_BC_horizontal : 2 * N_BC_horizontal][:, 1].unsqueeze(1)
    # col2 = out_neq_BC[N_BC_horizontal : 2 * N_BC_horizontal][:, 5].unsqueeze(1)
    # col3 = out_neq_BC[N_BC_horizontal : 2 * N_BC_horizontal][:, 8].unsqueeze(1)
    # f_neq_right = torch.cat((col1, col2, col3), dim=1)
    # loss_right = loss(f_neq_right, f_neq)

    # # 对于上边界只计算i = [2, 5, 6]的点
    # f_eq = torch.zeros(out_up.shape[0], 3)
    # f_neq = torch.zeros(out_up.shape[0], 3)
    # temp = 0
    # range = [2, 5, 6]
    # for i in range:
    #     f_eq[:, temp] = (
    #         D2Q9_Model.w[i]
    #         * rho_exact[2 * N_BC_horizontal : 2 * N_BC_horizontal + N_BC_vertical]
    #         * (
    #             1
    #             + (
    #                 D2Q9_Model.xi[i][0]
    #                 * u_exact[2 * N_BC_horizontal : 2 * N_BC_horizontal + N_BC_vertical]
    #                 + D2Q9_Model.xi[i][1]
    #                 * v_exact[2 * N_BC_horizontal : 2 * N_BC_horizontal + N_BC_vertical]
    #             )
    #             / RT
    #             + (
    #                 D2Q9_Model.xi[i][0]
    #                 * u_exact[2 * N_BC_horizontal : 2 * N_BC_horizontal + N_BC_vertical]
    #                 + D2Q9_Model.xi[i][1]
    #                 * v_exact[2 * N_BC_horizontal : 2 * N_BC_horizontal + N_BC_vertical]
    #             )
    #             ** 2
    #             / (2 * RT**2)
    #             - (
    #                 u_exact[2 * N_BC_horizontal : 2 * N_BC_horizontal + N_BC_vertical]
    #                 ** 2
    #                 + v_exact[2 * N_BC_horizontal : 2 * N_BC_horizontal + N_BC_vertical]
    #                 ** 2
    #             )
    #             / (2 * RT)
    #         )
    #     )
    #     temp += 1
    # f_neq = (
    #     -tau
    #     * (
    #         D2Q9_Model.xi[i][0] * gradients(f_eq, in_up[:, 0])
    #         + D2Q9_Model.xi[i][1] * gradients(f_eq, in_up[:, 1])
    #     )
    #     * 100000
    # )  # 将F_neq放大至O(1)量级
    # col1 = out_neq_BC[2 * N_BC_horizontal : 2 * N_BC_horizontal + N_BC_vertical][
    #     :, 2
    # ].unsqueeze(1)
    # col2 = out_neq_BC[2 * N_BC_horizontal : 2 * N_BC_horizontal + N_BC_vertical][
    #     :, 5
    # ].unsqueeze(1)
    # col3 = out_neq_BC[2 * N_BC_horizontal : 2 * N_BC_horizontal + N_BC_vertical][
    #     :, 6
    # ].unsqueeze(1)
    # f_neq_up = torch.cat((col1, col2, col3), dim=1)
    # loss_up = loss(f_neq_up, f_neq)

    # # 对于下边界只计算i = [4, 7, 8]的点
    # f_eq = torch.zeros(out_down.shape[0], 3)
    # f_neq = torch.zeros(out_down.shape[0], 3)
    # temp = 0
    # range = [4, 7, 8]
    # for i in range:
    #     f_eq[:, temp] = (
    #         D2Q9_Model.w[i]
    #         * rho_exact[2 * N_BC_horizontal + N_BC_vertical :]
    #         * (
    #             1
    #             + (
    #                 D2Q9_Model.xi[i][0] * u_exact[2 * N_BC_horizontal + N_BC_vertical :]
    #                 + D2Q9_Model.xi[i][1]
    #                 * v_exact[2 * N_BC_horizontal + N_BC_vertical :]
    #             )
    #             / RT
    #             + (
    #                 D2Q9_Model.xi[i][0] * u_exact[2 * N_BC_horizontal + N_BC_vertical :]
    #                 + D2Q9_Model.xi[i][1]
    #                 * v_exact[2 * N_BC_horizontal + N_BC_vertical :]
    #             )
    #             ** 2
    #             / (2 * RT**2)
    #             - (
    #                 u_exact[2 * N_BC_horizontal + N_BC_vertical :] ** 2
    #                 + v_exact[2 * N_BC_horizontal + N_BC_vertical :] ** 2
    #             )
    #             / (2 * RT)
    #         )
    #     )
    #     temp += 1
    # f_neq = (
    #     -tau
    #     * (
    #         D2Q9_Model.xi[i][0] * gradients(f_eq, in_down[:, 0])
    #         + D2Q9_Model.xi[i][1] * gradients(f_eq, in_down[:, 1])
    #     )
    #     * 100000
    # )  # 将F_neq放大至O(1)量级
    # col1 = out_neq_BC[2 * N_BC_horizontal + N_BC_vertical :][:, 4].unsqueeze(1)
    # col2 = out_neq_BC[2 * N_BC_horizontal + N_BC_vertical :][:, 7].unsqueeze(1)
    # col3 = out_neq_BC[2 * N_BC_horizontal + N_BC_vertical :][:, 8].unsqueeze(1)
    # f_neq_down = torch.cat((col1, col2, col3), dim=1)
    # loss_down = loss(f_neq_down, f_neq)

    # loss_fBC = loss_left + loss_right + loss_up + loss_down

    # """Calculate L_EBC"""
    # # 对于左边界只计算i = [1, 5, 8]的点
    # range = [1, 5, 8]
    # temp = 0
    # R = torch.zeros(N_BC_horizontal, 3)
    # cond = torch.zeros(N_BC_horizontal, 3)
    # for i in range:
    #     R[:, temp] = (
    #         D2Q9_Model.xi[i][0] * gradients(out_BC[:N_BC_horizontal, i], in_left[:, 0])
    #         + D2Q9_Model.xi[i][1]
    #         * gradients(out_BC[:N_BC_horizontal, i], in_left[:, 1])
    #         + 1 / tau * out_neq_BC[:N_BC_horizontal, i] / 100000
    #     )
    # loss_left = loss(R, cond)

    # # 对于右边界只计算i = [3, 6, 7]的点
    # range = [3, 6, 7]
    # temp = 0
    # R = torch.zeros(N_BC_horizontal, 3)
    # cond = torch.zeros(N_BC_horizontal, 3)
    # for i in range:
    #     R[:, temp] = (
    #         D2Q9_Model.xi[i][0]
    #         * gradients(
    #             out_BC[N_BC_horizontal : 2 * N_BC_horizontal, i], in_right[:, 0]
    #         )
    #         + D2Q9_Model.xi[i][1]
    #         * gradients(
    #             out_BC[N_BC_horizontal : 2 * N_BC_horizontal, i], in_right[:, 1]
    #         )
    #         + 1 / tau * out_neq_BC[N_BC_horizontal : 2 * N_BC_horizontal, i] / 100000
    #     )
    # loss_right = loss(R, cond)

    # # 对于上边界只计算i = [4, 7, 8]的点
    # range = [4, 7, 8]
    # temp = 0
    # R = torch.zeros(N_BC_vertical, 3)
    # cond = torch.zeros(N_BC_vertical, 3)
    # for i in range:
    #     R[:, temp] = (
    #         D2Q9_Model.xi[i][0]
    #         * gradients(
    #             out_BC[2 * N_BC_horizontal : 2 * N_BC_horizontal + N_BC_vertical, i],
    #             in_up[:, 0],
    #         )
    #         + D2Q9_Model.xi[i][1]
    #         * gradients(
    #             out_BC[2 * N_BC_horizontal : 2 * N_BC_horizontal + N_BC_vertical, i],
    #             in_up[:, 1],
    #         )
    #         + 1
    #         / tau
    #         * out_neq_BC[2 * N_BC_horizontal : 2 * N_BC_horizontal + N_BC_vertical, i]
    #         / 100000
    #     )
    # loss_up = loss(R, cond)

    # # 对于上边界只计算i = [2, 5, 6]的点
    # range = [2, 5, 6]
    # temp = 0
    # R = torch.zeros(N_BC_vertical, 3)
    # cond = torch.zeros(N_BC_vertical, 3)
    # for i in range:
    #     R[:, temp] = (
    #         D2Q9_Model.xi[i][0]
    #         * gradients(
    #             out_BC[2 * N_BC_horizontal + N_BC_vertical :, i],
    #             in_down[:, 0],
    #         )
    #         + D2Q9_Model.xi[i][1]
    #         * gradients(
    #             out_BC[2 * N_BC_horizontal + N_BC_vertical :, i],
    #             in_down[:, 1],
    #         )
    #         + 1 / tau * out_neq_BC[2 * N_BC_horizontal + N_BC_vertical :, i] / 100000
    #     )
    # loss_down = loss(R, cond)

    # loss_EBC = loss_left + loss_right + loss_up + loss_down

    return loss_rho + loss_rhou + loss_rhov  # + loss_fBC + loss_EBC


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


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


def train(in_var, model_eq, model_neq, loss_fn, optimizer_eq, optimizer_neq, fields):
    in_var.requires_grad = True
    optimizer_eq.zero_grad()
    optimizer_neq.zero_grad()
    out_eq = model_eq(in_var)
    out_neq = model_neq(in_var)
    residuals = CalResidualsLoss(in_var, out_eq, out_neq)
    loss_BC = CalBCLoss(out_eq, fields)

    loss_total = residuals + loss_BC
    loss_total.backward()
    pass


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    R = 8.314
    T = 200
    RT = R * T
    tau = 1.402485925e-5

    setup_seed(1234)
    N_internal = 2000
    N_BC_horizontal = 100
    N_BC_vertical = 200
    Box = [-0.5, 0, 0.5, 0.4]

    data = read_data()
    data = list(map(np.random.permutation, data))  # 打乱数据
    internal = data[0]
    left = data[1]
    right = data[2]
    up = data[3]
    down = data[4]
    # 在乱序数据下取前n个点以随机取点
    internal = internal[:N_internal]
    left = left[:N_BC_horizontal]
    right = right[:N_BC_horizontal]
    up = up[:N_BC_vertical]
    down = down[:N_BC_vertical]

    input = np.concatenate(
        [internal[:, 0:2], left[:, 0:2], right[:, 0:2], up[:, 0:2], down[:, 0:2]],
        axis=0,
    )
    field = np.concatenate(
        [internal[:, 2:], left[:, 2:], right[:, 2:], up[:, 2:], down[:, 2:]], axis=0
    )
    input = torch.tensor(input, dtype=torch.float32, device=device)
    field = torch.tensor(field, dtype=torch.float32, device=device)

    Model_eq = Net(planes=[2] + 6 * [40] + [9]).to(device)
    Model_neq = Net(planes=[2] + 6 * [40] + [9]).to(device)
    Optimizer_eq = torch.optim.Adam(params=Model_eq.parameters(), lr=0.001)
    Optimizer_neq = torch.optim.Adam(params=Model_neq.parameters(), lr=0.001)

    star_time = time.time()

    _tqdm = tqdm(range(100000))
    for i in _tqdm:
        input.requires_grad = True
        Optimizer_eq.zero_grad()
        Optimizer_neq.zero_grad()
        out_eq = Model_eq(input)
        out_neq = Model_neq(input)
        residuals = CalResidualsLoss(input, out_eq, out_neq)
        loss_BC = CalBCLoss(input, out_eq, out_neq, field)
        loss_total = residuals + loss_BC
        loss_total.backward()
        Optimizer_eq.step()
        Optimizer_neq.step()
        _tqdm.set_postfix(loss="{:.10f}".format(loss_total.item()))

    print("Time:", time.time() - star_time)
    print("Final loss:", loss_total.item())

    torch.save(Model_eq, "./resualt/Model_eq.pth")
    torch.save(Model_neq, "./resualt/Model_neq.pth")
