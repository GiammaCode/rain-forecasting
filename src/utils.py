import matplotlib.pyplot as plt

def plot_results(train_data, test_data, predictions, forecast, model_name):
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