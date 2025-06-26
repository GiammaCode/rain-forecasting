"""
Modulo per la Rete Neurale per il forecasting delle piogge settimanali
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from preprocessing import create_dataset, forecast_accuracy


class RainForecastNN(nn.Module):
    """
    Rete Neurale per il forecasting delle piogge
    Architettura: Input -> Hidden Layers -> Output
    """

    def __init__(self, input_size=52):
        super(RainForecastNN, self).__init__()

        # Architettura della rete (seguendo l'esempio originale ma adattata)
        self.network = nn.Sequential(
            nn.Linear(input_size, 26),  # input_size -> 26 neuroni
            nn.ReLU(),
            nn.Linear(26, 13),  # 26 -> 13 neuroni
            nn.ReLU(),
            nn.Linear(13, 6),  # 13 -> 6 neuroni
            nn.ReLU(),
            nn.Linear(6, 1)  # 6 -> 1 output (previsione)
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
    look_back = 52  # Usa 52 settimane (1 anno) per predire la settimana successiva

    # Creazione del dataset per supervised learning
    X, y = create_dataset(train_data, look_back)

    print(f"Dataset creato: X shape = {X.shape}, y shape = {y.shape}")
    print(f"Usando {look_back} settimane per predire la settimana successiva")

    # Conversione in tensori PyTorch
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)  # Aggiungi dimensione per compatibilità

    print("Inizializzazione e addestramento della Rete Neurale...")

    # Creazione del modello
    model = RainForecastNN(input_size=look_back)
    print(f"Architettura del modello:\n{model}")

    # Definizione della loss function e optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Learning rate leggermente più alto

    # Parametri di training
    n_epochs = 1000
    batch_size = 8  # Batch size più grande per stabilità

    print(f"Parametri training: {n_epochs} epoche, batch size = {batch_size}")
    print("Inizio training...")

    # Training loop
    model.train()
    epoch_losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Training in mini-batches
        for i in range(0, len(X_tensor), batch_size):
            # Estrazione batch
            X_batch = X_tensor[i:i + batch_size]
            y_batch = y_tensor[i:i + batch_size]

            # Forward pass
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_epoch_loss = epoch_loss / n_batches
        epoch_losses.append(avg_epoch_loss)

        # Stampa progresso ogni 100 epoche
        if (epoch + 1) % 100 == 0:
            print(f'Epoca {epoch + 1}/{n_epochs}, Loss: {avg_epoch_loss:.6f}')

    print("Training completato!")

    # Plot della loss durante il training
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epoch_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoca')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)

    # Plot ultimi 100 epoche per vedere convergenza
    plt.subplot(1, 2, 2)
    plt.plot(epoch_losses[-100:])
    plt.title('Training Loss (Ultime 100 epoche)')
    plt.xlabel('Epoca')
    plt.ylabel('MSE Loss')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Generazione previsioni in-sample
    print("Generazione previsioni in-sample...")
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_tensor).numpy().flatten()

    # Forecasting ricorsivo per i dati di test
    print("Generazione forecast out-of-sample con approccio ricorsivo...")

    # Inizializza con l'ultima finestra di training
    input_sequence = train_data[-look_back:].tolist()
    forecast_values = []

    model.eval()
    with torch.no_grad():
        for step in range(len(test_data)):
            # Converte la sequenza corrente in tensore
            input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0)

            # Genera previsione
            prediction = model(input_tensor).item()
            forecast_values.append(prediction)

            # Aggiorna la sequenza: rimuovi il primo elemento e aggiungi la previsione
            input_sequence = input_sequence[1:] + [prediction]

            if (step + 1) % 10 == 0:
                print(f"Generata previsione per settimana {step + 1}/52")

    forecast_values = np.array(forecast_values)

    # Visualizzazione risultati
    plot_nn_results(train_data, test_data, train_predictions, forecast_values, look_back)

    # Calcolo accuratezza
    accuracy = forecast_accuracy(forecast_values, test_data)
    print_nn_accuracy_results(accuracy)

    return train_predictions, forecast_values


def plot_nn_results(train_data, test_data, predictions, forecast, look_back):
    """
    Visualizza i risultati della Rete Neurale

    Args:
        train_data (np.array): Dati di training
        test_data (np.array): Dati di test reali
        predictions (np.array): Previsioni in-sample
        forecast (np.array): Previsioni out-of-sample
        look_back (int): Numero di time steps usati come input
    """

    plt.figure(figsize=(15, 10))

    # Plot 1: Previsioni in-sample
    plt.subplot(2, 1, 1)
    plt.plot(train_data, label="Dati Reali Training", alpha=0.7)
    # Le previsioni iniziano dopo look_back settimane
    pred_x = range(look_back, len(train_data))
    plt.plot(pred_x, predictions, label="Previsioni In-Sample", alpha=0.8)
    plt.title('Rete Neurale - Previsioni In-Sample (Training Data)')
    plt.xlabel('Settimane')
    plt.ylabel('Pioggia (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Forecast out-of-sample
    plt.subplot(2, 1, 2)
    # Mostra le ultime 52 settimane del training per contesto
    context_weeks = min(52, len(train_data))
    plt.plot(range(-context_weeks, 0), train_data[-context_weeks:],
             'k-', label='Training (Contesto)', alpha=0.7)
    plt.plot(range(len(test_data)), test_data, 'ko-',
             label='Dati Reali 2024', markersize=4)
    plt.plot(range(len(forecast)), forecast, 'g--',
             label='Forecast Neural Network', linewidth=2)

    plt.axvline(x=0, color='red', linestyle=':', alpha=0.5, label='Inizio Forecast')
    plt.title('Rete Neurale - Forecast Out-of-Sample (2024)')
    plt.xlabel('Settimane dal 2024')
    plt.ylabel('Pioggia (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_nn_accuracy_results(accuracy):
    """
    Stampa le metriche di accuratezza per la Rete Neurale

    Args:
        accuracy (dict): Dizionario con le metriche
    """

    print(f"\n--- RISULTATI ACCURATEZZA RETE NEURALE ---")
    print(f"MAPE (Mean Absolute Percentage Error): {accuracy['mape']:.4f} ({accuracy['mape'] * 100:.2f}%)")
    print(f"ME (Mean Error): {accuracy['me']:.4f}")
    print(f"MAE (Mean Absolute Error): {accuracy['mae']:.4f}")
    print(f"MPE (Mean Percentage Error): {accuracy['mpe']:.4f} ({accuracy['mpe'] * 100:.2f}%)")
    print(f"RMSE (Root Mean Square Error): {accuracy['rmse']:.4f}")
    print(f"Correlazione: {accuracy['corr']:.4f}")
    print("-" * 50)