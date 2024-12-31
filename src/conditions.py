import torch


def PDEs(x, y, device):
    condition = torch.zeros(x.shape[0], 1).to(device)
    return x.requires_grad_(True), y.requires_grad_(True), condition


def left(x, y):
    return x.requires_grad_(True), y.requires_grad_(True)


def right(x, y):
    return x.requires_grad_(True), y.requires_grad_(True)


def up(x, y):
    return x.requires_grad_(True), y.requires_grad_(True)


def down(x, y):
    return x.requires_grad_(True), y.requires_grad_(True)
