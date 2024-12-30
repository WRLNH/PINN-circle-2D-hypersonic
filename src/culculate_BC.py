import numpy as np
import pandas as pd

data = pd.read_csv('./data/filtered_data.csv', header=None)
data = data.to_numpy()

data_left = []
data_right = []
data_up = []
data_down = []

for row in data:
    if row[0] <= -0.498:
        data_left.append(row)
    elif row[0] >= 0.498:
        data_right.append(row)
    elif row[1] >= 0.398:
        data_up.append(row)
    elif row[1] <= 0.001:
        data_down.append(row)

data_left = np.array(data_left)
data_right = np.array(data_right)
data_up = np.array(data_up)
data_down = np.array(data_down)
data_left[:, 0] = -0.5
data_right[:, 0] = 0.5
data_up[:, 1] = 0.4
data_down[:, 1] = 0

np.savetxt('./data/data_internal.csv', data, delimiter=',')
np.savetxt('./data/data_left.csv', data_left, delimiter=',')
np.savetxt('./data/data_right.csv', data_right, delimiter=',')
np.savetxt('./data/data_up.csv', data_up, delimiter=',')
np.savetxt('./data/data_down.csv', data_down, delimiter=',')