import numpy as np
import pandas as pd
from preprocess.feature_engineer import FeatureEngineering

from sklearn.preprocessing import MinMaxScaler
from lstm_model import LSTMpredictor

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

read_csv = False
check_years= 3
hidden_size = 50
n_steps = 10
learning_rate = 0.1



if read_csv:
    df = pd.read_csv("data/CF_physics_April2_23.csv")
    df.reset_index(inplace=True)
    feat_eng_obj  = FeatureEngineering(df)
    data = feat_eng_obj.feature_engineer()
    data.to_pickle("fusion_df.pkl", compression="zip")
else:
    data = pd.read_pickle("fusion_df.pkl", compression="zip")


scaler = MinMaxScaler(feature_range=(-1,1))
data["y"] = scaler.fit_transform(data[["QUANTILE_NORM"]])
df = data.pivot(index="SYSTEM", columns='YEAR', values='y')
x = df.to_numpy()
numpy_train = x.copy()
print(f"train+test shape = {x.shape}")
print(f"dataframe shape = {df.shape}")

train_input = torch.Tensor(x[:, :-check_years]) # 97, 999
train_target = torch.Tensor(x[:, 1:-check_years+1])


test_input = torch.Tensor(x[:, -check_years:-1]) # 3, 999
test_target = torch.Tensor(x[:, -check_years+1:])

print(f"train shape = {train_input.shape}, test_input {test_input.shape}")

model = LSTMpredictor(hidden_size)
criterion = nn.MSELoss()
optimizer = optim.LBFGS(model.parameters(), lr=learning_rate)


for i in range(n_steps):
    print("Step", i)


    def closure():
        optimizer.zero_grad()
        out = model(train_input)
        loss = criterion(out, train_target)
        print("loss", loss.item())
        loss.backward()
        return loss


    optimizer.step(closure)

    with torch.no_grad():
        future = 6
        pred = model(test_input, future=future)
        loss = criterion(pred[:, :-future], test_target)
        print("test loss", loss.item())
        y = pred.detach().numpy()

    plt.figure(figsize=(12, 6))

    plt.title(f"Step {i + 1}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    n = train_input.shape[1]


    def draw(train, y_i, color):
        plt.plot(np.arange(len(train)), train, color, linewidth=2.0)
        plt.plot(np.arange(n, n + future + check_years - 1), y_i[:], color + ":", linewidth=2.0)


    draw(x[4], y[4], "k")
    draw(x[5], y[5], "r")  # Ni_all
    draw(x[6], y[6], "g")
    draw(x[7], y[7], "b")
    draw(x[9], y[9], "m")
    draw(x[10], y[10], "y")

    plt.savefig("predict%d.pdf" % i)
    plt.close()



