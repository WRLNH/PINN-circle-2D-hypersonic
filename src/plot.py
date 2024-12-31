import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train import Net
from D2Q9 import D2Q9_Model

model = torch.load("./result/Model_eq.pth").to("cuda")
data = pd.read_csv("./data/filtered_data.csv", header=None)
data = data.to_numpy()

x = data[:, 0]
y = data[:, 1]
n_exact = data[:, 2]
nrho_exact = data[:, 3]
rho_exact = data[:, 4]
u_exact = data[:, 5]
v_exact = data[:, 6]
temp_exact = data[:, 7]
press_exact = data[:, 8]
rhou_exact = rho_exact * u_exact
rhov_exact = rho_exact * v_exact

xc = torch.tensor(x, dtype=torch.float32).to(device="cuda").unsqueeze(1)
yc = torch.tensor(y, dtype=torch.float32).to(device="cuda").unsqueeze(1)
output = model(torch.cat([xc, yc], dim=1))
rho = torch.sum(output, dim=1).detach().cpu().numpy()
rhou = torch.zeros_like(output)
rhov = torch.zeros_like(output)
for i in range(0, 9):
    rhou[:, i] = output[:, i] * D2Q9_Model.xi[i][0]
    rhov[:, i] = output[:, i] * D2Q9_Model.xi[i][1]
rhou = torch.sum(rhou, dim=1).detach().cpu().numpy()
rhov = torch.sum(rhov, dim=1).detach().cpu().numpy()
u = rhou / rho
v = rhov / rho

plt.subplot(2, 1, 1)
plt.scatter(x, y, c=rho, s=1, cmap="rainbow")
plt.colorbar()
plt.axis("equal")

plt.subplot(2, 1, 2)
plt.scatter(x, y, c=rho_exact, s=1, cmap="rainbow")
plt.colorbar()
plt.axis("equal")
plt.show()
