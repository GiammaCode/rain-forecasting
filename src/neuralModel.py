# neuralModel.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class NeuralForecaster(nn.Module):
    def __init__(self, input_size=38):
        super(NeuralForecaster, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 24),
            nn.ReLU(),
            nn.Linear(24, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
            nn.Linear(6, 1)
        )

    def forward(self, x):
        return self.network(x)


def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back])
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)


def train_model(model, X, y, lr=0.0007, n_epochs=1000, batch_size=1):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i + batch_size]
            ybatch = y[i:i + batch_size]
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')
    return model


def recursive_forecast(model, initial_input, n_steps):
    model.eval()
    input_seq = initial_input.tolist()
    forecast = []

    with torch.no_grad():
        for _ in range(n_steps):
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
            pred = model(input_tensor).item()
            forecast.append(pred)
            input_seq = input_seq[1:] + [pred]
    return forecast


def plot_results(train, train_pred, forecast, look_back, test):
    plt.figure()
    plt.plot(train, label="Actual")
    plt.plot(range(look_back, look_back + len(train_pred)), train_pred, label="Train Prediction")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(train[-look_back:], label="Last train window")
    plt.plot(forecast, "-o", label="Forecast")
    plt.legend()
    plt.grid()
    plt.title("Neural Network Forecast")
    plt.show()


# Optional: evaluation metric
def forecast_accuracy(forecast, actual):
    forecast = np.array(forecast)
    actual = np.array(actual)
    mae = np.mean(np.abs(forecast - actual))
    mse = np.mean((forecast - actual) ** 2)
    rmse = np.sqrt(mse)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}
