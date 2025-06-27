"""
Modulo per la Rete Neurale per il forecasting delle piogge settimanali
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from preprocessing import create_dataset, forecast_accuracy
from utils import *


class RainForecastNN(nn.Module):
    """
    Rete Neurale per il forecasting delle piogge
    """
    def __init__(self, input_size=104):
        super(RainForecastNN, self).__init__()
        self.input_size = input_size
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


def set_seed(seed=42):
    """Imposta il seed per riproducibilità"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, X, y, lr=0.0007, n_epochs=1000, batch_size=8, patience=100):
    """Addestra il modello"""
    loss_fn = nn.HuberLoss(delta=3.0)  # delta si può regolare
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_losses = []
    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X), batch_size):
            Xbatch = X[i:i + batch_size]
            ybatch = y[i:i + batch_size]
            y_pred = model(Xbatch)
            loss = loss_fn(y_pred.squeeze(1), ybatch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_epoch_loss = epoch_loss / n_batches
        epoch_losses.append(avg_epoch_loss)

        # Early stopping check (commentato come nell'originale)
        # if avg_epoch_loss < best_loss - 1e-5:
        #     best_loss = avg_epoch_loss
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f"Early stopping at epoch {epoch}, best loss: {best_loss:.6f}")
        #         break

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f'Epoch {epoch}, Loss: {avg_epoch_loss:.6f}')

    return model, epoch_losses


def recursive_forecast(model, initial_input, n_steps):
    """Genera forecast ricorsivo"""
    model.eval()
    input_seq = initial_input.tolist()
    forecast = []

    with torch.no_grad():
        for step in range(n_steps):
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
            pred = model(input_tensor).item()
            forecast.append(pred)
            input_seq = input_seq[1:] + [pred]

            if (step + 1) % 10 == 0:
                print(f"Generata previsione per settimana {step + 1}/{n_steps}")

    return forecast


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
    look_back = 104  # Usa 104 settimane (2 anni) per predire la settimana successiva

    # Imposta seed per riproducibilità
    set_seed(42)

    # Creazione del dataset per supervised learning
    X, y = create_dataset(train_data, look_back)

    print(f"Dataset creato: X shape = {X.shape}, y shape = {y.shape}")
    print(f"Usando {look_back} settimane per predire la settimana successiva")

    # Conversione in tensori PyTorch
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    print("Inizializzazione e addestramento della Rete Neurale...")

    # Creazione del modello
    model = RainForecastNN(input_size=look_back)
    print(f"Architettura del modello:\n{model}")

    # Training del modello
    print("Inizio training...")
    model, epoch_losses = train_model(
        model, X_tensor, y_tensor,
        lr=0.0007,
        n_epochs=1000,
        batch_size=8,
        patience=100
    )
    print("Training completato!")

    # Plot della loss durante il training
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoca')
    plt.ylabel('Huber Loss')
    plt.grid(True, alpha=0.3)

    # Plot ultimi 100 epoche per vedere convergenza
    plt.subplot(1, 2, 2)
    plt.plot(epoch_losses[-100:])
    plt.title('Training Loss (Ultime 100 epoche)')
    plt.xlabel('Epoca')
    plt.ylabel('Huber Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Generazione previsioni in-sample
    print("Generazione previsioni in-sample...")
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_tensor).squeeze().numpy()

    # Forecasting ricorsivo per i dati di test
    print("Generazione forecast out-of-sample con approccio ricorsivo...")

    # Inizializza con l'ultima finestra di training
    initial_input = train_data[-look_back:]
    forecast_values = recursive_forecast(model, initial_input, len(test_data))
    forecast_values = np.array(forecast_values)

    # Visualizzazione risultati
    plot_results(train_data, test_data, train_predictions, forecast_values, "RETE NEURALE")

    # Calcolo accuratezza
    accuracy = forecast_accuracy(forecast_values, test_data)
    print_accuracy_results(accuracy, "RETE NEURALE")

    return train_predictions, forecast_values
