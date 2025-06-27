import matplotlib.pyplot as plt
import numpy as np

def plot_results(train_data, test_data, predictions, forecast, model_name, look_back=0):
    """
       Visualizza i risultati dei modelli

       Args:
           train_data (np.array): Dati di training
           test_data (np.array): Dati di test reali
           predictions (np.array): Previsioni in-sample
           forecast (np.array): Previsioni out-of-sample
       """

    plt.figure(figsize=(15, 10))

    # Plot 1: Previsioni in-sample
    plt.subplot(2, 1, 1)
    plt.plot(train_data, label="Dati Reali Training", alpha=0.7)
    if look_back > 0:
        pred_x = range(look_back, len(train_data))
        plt.plot(pred_x, predictions, label="Previsioni In-Sample", alpha=0.8)
    else:
        plt.plot(predictions, label="Previsioni In-Sample", alpha=0.8)
    plt.title(f'{model_name.upper()} - Previsioni In-Sample (Training Data)')
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
    plt.plot(range(len(forecast)), forecast, 'r--',
             label= f'Forecast {model_name.upper()}', linewidth=2)

    plt.axvline(x=0, color='red', linestyle=':', alpha=0.5, label='Inizio Forecast')
    plt.title(f'{model_name.upper()} - Forecast Out-of-Sample (2024)')
    plt.xlabel('Settimane dal 2024')
    plt.ylabel('Pioggia (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def forecast_accuracy(forecast, actual):
    """
    Calcola metriche di accuratezza per le previsioni

    Args:
        forecast (np.array): Valori previsti
        actual (np.array): Valori reali

    Returns:
        dict: Dizionario con le metriche di accuratezza
    """
    # Maschera per evitare divisione per zero
    non_zero_mask = actual != 0

    if np.sum(non_zero_mask) > 0:
        # MAPE solo su valori non-zero
        mape = np.mean(np.abs(forecast[non_zero_mask] - actual[non_zero_mask]) /
                       np.abs(actual[non_zero_mask]))
        # MPE solo su valori non-zero
        mpe = np.mean((forecast[non_zero_mask] - actual[non_zero_mask]) /
                      actual[non_zero_mask])
    else:
        mape = np.nan
        mpe = np.nan

    # Altre metriche rimangono invariate
    me = np.mean(forecast - actual)
    mae = np.mean(np.abs(forecast - actual))
    rmse = np.mean((forecast - actual) ** 2) ** 0.5
    corr = np.corrcoef(forecast, actual)[0, 1]

    return {
        'mape': mape,
        'me': me,
        'mae': mae,
        'mpe': mpe,
        'rmse': rmse,
        'corr': corr,
        'zero_weeks_actual': np.sum(actual == 0),
        'zero_weeks_forecast': np.sum(forecast == 0)
    }

def print_accuracy_results(accuracy, model_name):
    """
    Stampa le metriche di accuratezza in formato leggibile

    Args:
        accuracy (dict): Dizionario con le metriche
        model_name (str): Nome del modello
    """

    print(f"\n--- RISULTATI ACCURATEZZA {model_name.upper()} ---")
    print(f"MAPE (Mean Absolute Percentage Error): {accuracy['mape']:.4f} ({accuracy['mape']*100:.2f}%)")
    print(f"ME (Mean Error): {accuracy['me']:.4f}")
    print(f"MAE (Mean Absolute Error): {accuracy['mae']:.4f}")
    print(f"MPE (Mean Percentage Error): {accuracy['mpe']:.4f} ({accuracy['mpe']*100:.2f}%)")
    print(f"RMSE (Root Mean Square Error): {accuracy['rmse']:.4f}")
    print(f"Correlazione: {accuracy['corr']:.4f}")
    print("-" * 50)

def create_dataset(data, look_back=1):
    """
    Converte una serie temporale in un dataset per supervised learning
    Stessa funzione dell'esempio originale

    Args:
        data (np.array): Serie temporale
        look_back (int): Numero di time steps da usare come input

    Returns:
        tuple: (X, y) array per training
    """

    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])

    return np.array(X), np.array(y)