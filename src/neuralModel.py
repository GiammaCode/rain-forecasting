# neuralModel.py

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class NeuralForecaster(nn.Module):
    def __init__(self, input_size=52):
        super(NeuralForecaster, self).__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(52, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.network(x)


def train_model(model, X, y, lr=0.0007, n_epochs=1000, batch_size=1):
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i + batch_size]
            ybatch = y[i:i + batch_size]
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred.squeeze(1), ybatch)
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

def plot_results(forecast, test):
    plt.figure()
    x_forecast = list(range(1, len(forecast)+1))  # Settimane 1–52 del 2024
    plt.plot(x_forecast, test, label="Actual 2024")
    plt.plot(x_forecast, forecast, label="Forecast 2024")
    plt.legend()
    plt.grid()
    plt.title("Neural Network Forecast - 2024")
    plt.xlabel("Settimana del 2024")
    plt.ylabel("Pioggia (mm)")
    plt.show()
