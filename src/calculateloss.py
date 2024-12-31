import torch
import conditions

from D2Q9 import D2Q9_Model

loss_fn = torch.nn.MSELoss()
tau = 1.402485925e-5


def gradients(fn, x, order=1):
    if order == 1:
        return torch.autograd.grad(
            fn,
            x,
            grad_outputs=torch.ones_like(fn),
            create_graph=True,
            only_inputs=True,
        )[0]
    else:
        return gradients(gradients(fn, x), x, order=order - 1)


def loss_PDEs(model_eq, model_neq, x, y, device):
    x, y, condition = conditions.PDEs(x, y, device)
    outputs_eq = model_eq(torch.cat([x, y], dim=1))
    outputs_neq = model_neq(torch.cat([x, y], dim=1)) / 100000
    f = outputs_eq + outputs_neq
    f_0 = f[:, 0].reshape(x.shape[0], 1)
    f_1 = f[:, 1].reshape(x.shape[0], 1)
    f_2 = f[:, 2].reshape(x.shape[0], 1)
    f_3 = f[:, 3].reshape(x.shape[0], 1)
    f_4 = f[:, 4].reshape(x.shape[0], 1)
    f_5 = f[:, 5].reshape(x.shape[0], 1)
    f_6 = f[:, 6].reshape(x.shape[0], 1)
    f_7 = f[:, 7].reshape(x.shape[0], 1)
    f_8 = f[:, 8].reshape(x.shape[0], 1)
    f_eq_0 = outputs_eq[:, 0].reshape(x.shape[0], 1)
    f_eq_1 = outputs_eq[:, 1].reshape(x.shape[0], 1)
    f_eq_2 = outputs_eq[:, 2].reshape(x.shape[0], 1)
    f_eq_3 = outputs_eq[:, 3].reshape(x.shape[0], 1)
    f_eq_4 = outputs_eq[:, 4].reshape(x.shape[0], 1)
    f_eq_5 = outputs_eq[:, 5].reshape(x.shape[0], 1)
    f_eq_6 = outputs_eq[:, 6].reshape(x.shape[0], 1)
    f_eq_7 = outputs_eq[:, 7].reshape(x.shape[0], 1)
    f_eq_8 = outputs_eq[:, 8].reshape(x.shape[0], 1)
    f_neq_0 = outputs_neq[:, 0].reshape(x.shape[0], 1)
    f_neq_1 = outputs_neq[:, 1].reshape(x.shape[0], 1)
    f_neq_2 = outputs_neq[:, 2].reshape(x.shape[0], 1)
    f_neq_3 = outputs_neq[:, 3].reshape(x.shape[0], 1)
    f_neq_4 = outputs_neq[:, 4].reshape(x.shape[0], 1)
    f_neq_5 = outputs_neq[:, 5].reshape(x.shape[0], 1)
    f_neq_6 = outputs_neq[:, 6].reshape(x.shape[0], 1)
    f_neq_7 = outputs_neq[:, 7].reshape(x.shape[0], 1)
    f_neq_8 = outputs_neq[:, 8].reshape(x.shape[0], 1)
    return (
        loss_fn(
            D2Q9_Model.xi[0][0] * gradients(f_0, x, 1)
            + D2Q9_Model.xi[0][1] * gradients(f_0, y, 1)
            + 1 / tau * f_neq_0,
            condition,
        )
        + loss_fn(
            D2Q9_Model.xi[1][0] * gradients(f_1, x, 1)
            + D2Q9_Model.xi[1][1] * gradients(f_1, y, 1)
            + 1 / tau * f_neq_1,
            condition,
        )
        + loss_fn(
            D2Q9_Model.xi[2][0] * gradients(f_2, x, 1)
            + D2Q9_Model.xi[2][1] * gradients(f_2, y, 1)
            + 1 / tau * f_neq_2,
            condition,
        )
        + loss_fn(
            D2Q9_Model.xi[3][0] * gradients(f_3, x, 1)
            + D2Q9_Model.xi[3][1] * gradients(f_3, y, 1)
            + 1 / tau * f_neq_3,
            condition,
        )
        + loss_fn(
            D2Q9_Model.xi[4][0] * gradients(f_4, x, 1)
            + D2Q9_Model.xi[4][1] * gradients(f_4, y, 1)
            + 1 / tau * f_neq_4,
            condition,
        )
        + loss_fn(
            D2Q9_Model.xi[5][0] * gradients(f_5, x, 1)
            + D2Q9_Model.xi[5][1] * gradients(f_5, y, 1)
            + 1 / tau * f_neq_5,
            condition,
        )
        + loss_fn(
            D2Q9_Model.xi[6][0] * gradients(f_6, x, 1)
            + D2Q9_Model.xi[6][1] * gradients(f_6, y, 1)
            + 1 / tau * f_neq_6,
            condition,
        )
        + loss_fn(
            D2Q9_Model.xi[7][0] * gradients(f_7, x, 1)
            + D2Q9_Model.xi[7][1] * gradients(f_7, y, 1)
            + 1 / tau * f_neq_7,
            condition,
        )
        + loss_fn(
            D2Q9_Model.xi[8][0] * gradients(f_8, x, 1)
            + D2Q9_Model.xi[8][1] * gradients(f_8, y, 1)
            + 1 / tau * f_neq_8,
            condition,
        )
    )


def loss_BC_left(model_eq, model_neq, x, y, rho, u, v, temp, press):
    x, y = conditions.left(x, y)
    outputs_eq = model_eq(torch.cat([x, y], dim=1))
    outputs_neq = model_neq(torch.cat([x, y], dim=1)) / 100000
    rho_pred = torch.sum(outputs_eq, dim=1)
    rhou = (
        D2Q9_Model.xi[0][0] * outputs_eq[:, 0]
        + D2Q9_Model.xi[1][0] * outputs_eq[:, 1]
        + D2Q9_Model.xi[2][0] * outputs_eq[:, 2]
        + D2Q9_Model.xi[3][0] * outputs_eq[:, 3]
        + D2Q9_Model.xi[4][0] * outputs_eq[:, 4]
        + D2Q9_Model.xi[5][0] * outputs_eq[:, 5]
        + D2Q9_Model.xi[6][0] * outputs_eq[:, 6]
        + D2Q9_Model.xi[7][0] * outputs_eq[:, 7]
        + D2Q9_Model.xi[8][0] * outputs_eq[:, 8]
    )
    rhov = (
        D2Q9_Model.xi[0][1] * outputs_eq[:, 0]
        + D2Q9_Model.xi[1][1] * outputs_eq[:, 1]
        + D2Q9_Model.xi[2][1] * outputs_eq[:, 2]
        + D2Q9_Model.xi[3][1] * outputs_eq[:, 3]
        + D2Q9_Model.xi[4][1] * outputs_eq[:, 4]
        + D2Q9_Model.xi[5][1] * outputs_eq[:, 5]
        + D2Q9_Model.xi[6][1] * outputs_eq[:, 6]
        + D2Q9_Model.xi[7][1] * outputs_eq[:, 7]
        + D2Q9_Model.xi[8][1] * outputs_eq[:, 8]
    )
    u_pred = rhou / rho_pred
    v_pred = rhov / rho_pred
    return loss_fn(rho_pred, rho) + loss_fn(u_pred, u) + loss_fn(v_pred, v)


def loss_BC_right(model_eq, model_neq, x, y, rho, u, v, temp, press):
    x, y = conditions.right(x, y)
    outputs_eq = model_eq(torch.cat([x, y], dim=1))
    outputs_neq = model_neq(torch.cat([x, y], dim=1)) / 100000
    rho_pred = torch.sum(outputs_eq, dim=1)
    rhou = (
        D2Q9_Model.xi[0][0] * outputs_eq[:, 0]
        + D2Q9_Model.xi[1][0] * outputs_eq[:, 1]
        + D2Q9_Model.xi[2][0] * outputs_eq[:, 2]
        + D2Q9_Model.xi[3][0] * outputs_eq[:, 3]
        + D2Q9_Model.xi[4][0] * outputs_eq[:, 4]
        + D2Q9_Model.xi[5][0] * outputs_eq[:, 5]
        + D2Q9_Model.xi[6][0] * outputs_eq[:, 6]
        + D2Q9_Model.xi[7][0] * outputs_eq[:, 7]
        + D2Q9_Model.xi[8][0] * outputs_eq[:, 8]
    )
    rhov = (
        D2Q9_Model.xi[0][1] * outputs_eq[:, 0]
        + D2Q9_Model.xi[1][1] * outputs_eq[:, 1]
        + D2Q9_Model.xi[2][1] * outputs_eq[:, 2]
        + D2Q9_Model.xi[3][1] * outputs_eq[:, 3]
        + D2Q9_Model.xi[4][1] * outputs_eq[:, 4]
        + D2Q9_Model.xi[5][1] * outputs_eq[:, 5]
        + D2Q9_Model.xi[6][1] * outputs_eq[:, 6]
        + D2Q9_Model.xi[7][1] * outputs_eq[:, 7]
        + D2Q9_Model.xi[8][1] * outputs_eq[:, 8]
    )
    u_pred = rhou / rho_pred
    v_pred = rhov / rho_pred
    return loss_fn(rho_pred, rho) + loss_fn(u_pred, u) + loss_fn(v_pred, v)


def loss_BC_up(model_eq, model_neq, x, y, rho, u, v, temp, press):
    x, y = conditions.up(x, y)
    outputs_eq = model_eq(torch.cat([x, y], dim=1))
    outputs_neq = model_neq(torch.cat([x, y], dim=1)) / 100000
    rho_pred = torch.sum(outputs_eq, dim=1)
    rhou = (
        D2Q9_Model.xi[0][0] * outputs_eq[:, 0]
        + D2Q9_Model.xi[1][0] * outputs_eq[:, 1]
        + D2Q9_Model.xi[2][0] * outputs_eq[:, 2]
        + D2Q9_Model.xi[3][0] * outputs_eq[:, 3]
        + D2Q9_Model.xi[4][0] * outputs_eq[:, 4]
        + D2Q9_Model.xi[5][0] * outputs_eq[:, 5]
        + D2Q9_Model.xi[6][0] * outputs_eq[:, 6]
        + D2Q9_Model.xi[7][0] * outputs_eq[:, 7]
        + D2Q9_Model.xi[8][0] * outputs_eq[:, 8]
    )
    rhov = (
        D2Q9_Model.xi[0][1] * outputs_eq[:, 0]
        + D2Q9_Model.xi[1][1] * outputs_eq[:, 1]
        + D2Q9_Model.xi[2][1] * outputs_eq[:, 2]
        + D2Q9_Model.xi[3][1] * outputs_eq[:, 3]
        + D2Q9_Model.xi[4][1] * outputs_eq[:, 4]
        + D2Q9_Model.xi[5][1] * outputs_eq[:, 5]
        + D2Q9_Model.xi[6][1] * outputs_eq[:, 6]
        + D2Q9_Model.xi[7][1] * outputs_eq[:, 7]
        + D2Q9_Model.xi[8][1] * outputs_eq[:, 8]
    )
    u_pred = rhou / rho_pred
    v_pred = rhov / rho_pred
    return loss_fn(rho_pred, rho) + loss_fn(u_pred, u) + loss_fn(v_pred, v)


def loss_BC_down(model_eq, model_neq, x, y, rho, u, v, temp, press):
    x, y = conditions.down(x, y)
    outputs_eq = model_eq(torch.cat([x, y], dim=1))
    outputs_neq = model_neq(torch.cat([x, y], dim=1)) / 100000
    rho_pred = torch.sum(outputs_eq, dim=1)
    rhou = (
        D2Q9_Model.xi[0][0] * outputs_eq[:, 0]
        + D2Q9_Model.xi[1][0] * outputs_eq[:, 1]
        + D2Q9_Model.xi[2][0] * outputs_eq[:, 2]
        + D2Q9_Model.xi[3][0] * outputs_eq[:, 3]
        + D2Q9_Model.xi[4][0] * outputs_eq[:, 4]
        + D2Q9_Model.xi[5][0] * outputs_eq[:, 5]
        + D2Q9_Model.xi[6][0] * outputs_eq[:, 6]
        + D2Q9_Model.xi[7][0] * outputs_eq[:, 7]
        + D2Q9_Model.xi[8][0] * outputs_eq[:, 8]
    )
    rhov = (
        D2Q9_Model.xi[0][1] * outputs_eq[:, 0]
        + D2Q9_Model.xi[1][1] * outputs_eq[:, 1]
        + D2Q9_Model.xi[2][1] * outputs_eq[:, 2]
        + D2Q9_Model.xi[3][1] * outputs_eq[:, 3]
        + D2Q9_Model.xi[4][1] * outputs_eq[:, 4]
        + D2Q9_Model.xi[5][1] * outputs_eq[:, 5]
        + D2Q9_Model.xi[6][1] * outputs_eq[:, 6]
        + D2Q9_Model.xi[7][1] * outputs_eq[:, 7]
        + D2Q9_Model.xi[8][1] * outputs_eq[:, 8]
    )
    u_pred = rhou / rho_pred
    v_pred = rhov / rho_pred
    return loss_fn(rho_pred, rho) + loss_fn(u_pred, u) + loss_fn(v_pred, v)
