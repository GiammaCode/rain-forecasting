"""
Modulo per la Rete Neurale per il forecasting delle piogge settimanali
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import *


class RainForecastNN(nn.Module):
    def __init__(self, input_size=104):
        super(RainForecastNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


def neural_network_forecast(train_data, test_data):
    """
    Addestra una rete neurale e genera previsioni

    Args:
        train_data (np.array): Dati di training (2014-2023)
        test_data (np.array): Dati di test (2024)

    Returns:
        tuple: (predictions_in_sample, forecast_out_sample)
    """

    print("Preparazione dati per la Rete Neurale...")

    # Parametri
    look_back = 104
    lr = 0.001
    epochs = 200
    batch_size = 16

    # Set seed per riproducibilità
    torch.manual_seed(60)
    np.random.seed(60)

    # Creazione dataset
    X, y = create_dataset(train_data, look_back)
    print(f"Dataset: X {X.shape}, y {y.shape}")

    # Normalizzazione
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    # Conversione in tensori
    X_tensor = torch.FloatTensor(X_scaled)
    y_tensor = torch.FloatTensor(y_scaled)

    # Creazione e training del modello
    print("Training della rete neurale...")
    model = RainForecastNN(input_size=look_back)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Mini-batch training
        for i in range(0, len(X_tensor), batch_size):
            X_batch = X_tensor[i:i + batch_size]
            y_batch = y_tensor[i:i + batch_size]

            # Forward pass
            y_pred = model(X_batch).squeeze()
            loss = loss_fn(y_pred, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 200 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {total_loss/len(X_tensor)*batch_size:.6f}')

    print("Training completato!")

    # Previsioni in-sample
    print("Generazione previsioni in-sample...")
    model.eval()
    with torch.no_grad():
        train_pred_scaled = model(X_tensor).squeeze().numpy()
        # Denormalizzazione
        train_predictions = scaler_y.inverse_transform(train_pred_scaled.reshape(-1, 1)).flatten()

    # Forecasting out-of-sample
    print("Generazione forecast out-of-sample...")
    model.eval()

    # Inizializza con ultimi 104 valori normalizzati
    last_sequence = scaler_X.transform(train_data[-look_back:].reshape(1, -1))[0]
    forecast_scaled = []

    with torch.no_grad():
        for step in range(len(test_data)):
            # Predizione
            input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0)
            pred_scaled = model(input_tensor).item()
            forecast_scaled.append(pred_scaled)

            # Update sequence (shift + nuovo valore)
            last_sequence = np.roll(last_sequence, -1)
            last_sequence[-1] = pred_scaled

            if (step + 1) % 10 == 0:
                print(f"Generata previsione {step + 1}/{len(test_data)}")

    # Denormalizzazione forecast
    forecast_values = scaler_y.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()

    # Visualizzazione e metriche
    plot_results(train_data, test_data, train_predictions, forecast_values, "RETE NEURALE", look_back)

    accuracy = forecast_accuracy(forecast_values, test_data)
    print_accuracy_results(accuracy, "RETE NEURALE")

    return train_predictions, forecast_values