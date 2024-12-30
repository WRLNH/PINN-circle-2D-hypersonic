import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

from basic_model import DeepModel_single, DeepModel_multi, gradients
from D2Q9 import D2Q9_Model

class Net(DeepModel_single):
    def __init__(self, planes):
        super(Net, self).__init__(planes, active=nn.Tanh())

def CalResidualsLoss(in_var, out_eq, out_neq):
    loss = nn.MSELoss()

    tau = 1.3978e-5
    f = out_eq + out_neq
    f_eq = out_eq
    f_neq = out_neq
    R = torch.zeros_like(f)
    dfda = gradients(f, in_var)
    dfdx, dfdy = dfda[..., 0:1], dfda[..., 1:2]
    for i in range(0, 9):
        R[:, i] = D2Q9_Model.xi[i][0] * dfdx[:, i] + D2Q9_Model.xi[i][1] * dfdy[:, i] + 1 / tau * f_neq[:, i]
    R = torch.sum(R, dim=1)
    cond = torch.zeros_like(R)
    return loss(R, cond)

def CalBCLoss(out_eq, fields):
    loss = nn.MSELoss()

    """Calculate L_mBC"""
    BC_out = out_eq[2000:2600]
    BC_exact = fields[2000:2600]
    rho_pred = torch.sum(BC_out, dim=1)
    rho_exact = BC_exact[:, 2]
    u_exact = BC_exact[:, 3]
    v_exact = BC_exact[:, 4]
    loss_rho = loss(rho_pred, rho_exact)

    rhou = torch.zeros_like(BC_out)
    rhov = torch.zeros_like(BC_out)
    for i in range(0, 9):
        rhou[:, i] = D2Q9_Model.xi[i][0] * BC_out[:, i]
        rhov[:, i] = D2Q9_Model.xi[i][1] * BC_out[:, i]
    rhou_pred = torch.sum(rhou, dim=1)
    rhov_pred = torch.sum(rhov, dim=1)
    loss_rhou = loss(rhou_pred, rho_exact * u_exact)
    loss_rhov = loss(rhov_pred, rho_exact * v_exact)

    """Calculate L_fBC"""



    return loss_rho + loss_rhou + loss_rhov

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def read_data():
    data_internal = pd.read_csv('./data/data_internal.csv', header=None)
    data_internal = data_internal.to_numpy()
    data_left = pd.read_csv('./data/data_left.csv', header=None)
    data_left = data_left.to_numpy()
    data_right = pd.read_csv('./data/data_right.csv', header=None)
    data_right = data_right.to_numpy()
    data_up = pd.read_csv('./data/data_up.csv', header=None)
    data_up = data_up.to_numpy()
    data_down = pd.read_csv('./data/data_down.csv', header=None)
    data_down = data_down.to_numpy()
    
    return data_internal, data_left, data_right, data_up, data_down

def train(in_var, model_eq, model_neq, loss_fn, optimizer_eq, optimizer_neq, fields):
    in_var.requires_grad = True
    optimizer_eq.zero_grad()
    optimizer_neq.zero_grad()
    out_eq = model_eq(in_var)
    out_neq = model_neq(in_var)
    residuals = CalResidualsLoss(in_var, out_eq, out_neq)
    loss_BC_phi = CalBCLoss(out_eq, fields)

    loss_total = residuals + loss_BC_phi
    loss_total.backward()
    pass

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    setup_seed(1234)
    N_internal = 2000
    N_BC_horizontal = 100
    N_BC_vertical = 200
    Box = [-0.5, 0, 0.5, 0.4]

    data = read_data()
    data = list(map(np.random.permutation, data)) # 打乱数据
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

    input = np.concatenate([internal[:, 0:2], left[:, 0:2], right[:, 0:2], up[:, 0:2], down[:, 0:2]], axis=0)
    field = np.concatenate([internal[:, 2:], left[:, 2:], right[:, 2:], up[:, 2:], down[:, 2:]], axis=0)
    input = torch.tensor(input, dtype=torch.float32, device=device)
    field = torch.tensor(field, dtype=torch.float32)

    Model_eq = Net(planes=[2] + 6 * [40] + [9], active=nn.Tanh()).to(device)
    Model_neq = Net(planes=[2] + 6 * [40] + [9], active=nn.Tanh()).to(device)
    Optimizer_eq = torch.optim.Adam(params=Model_eq.parameters(), lr=0.001)
    Optimizer_neq = torch.optim.Adam(params=Model_neq.parameters(), lr=0.001)

    star_time = time.time()

    for epoch in range(10000):
        pass