"""
Modulo per il modello SARIMA per il forecasting delle piogge settimanali
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from preprocessing import forecast_accuracy
import warnings
warnings.filterwarnings('ignore')

def sarima_forecast(train_data, test_data):
    """
    Addestra un modello SARIMA e genera previsioni

    Args:
        train_data (np.array): Dati di training (2014-2023)
        test_data (np.array): Dati di test (2024)

    Returns:
        tuple: (predictions_in_sample, forecast_out_sample)
    """

    print("Addestramento modello SARIMA...")
    print("Parametri: order=(1,1,1), seasonal_order=(1,1,1,52)")

    # Definizione parametri del modello
    # order = (p, d, q) - parametri non stagionali
    # seasonal_order = (P, D, Q, s) - parametri stagionali
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 52)  # stagionalità settimanale (52 settimane)

    try:
        # Creazione e addestramento del modello
        model = SARIMAX(train_data,
                       order=order,
                       seasonal_order=seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)

        print("Fitting del modello in corso...")
        fitted_model = model.fit(disp=False, maxiter=100)

        print("Modello addestrato con successo!")
        print(f"AIC: {fitted_model.aic:.2f}")
        print(f"BIC: {fitted_model.bic:.2f}")

        # Previsioni in-sample (sui dati di training)
        print("Generazione previsioni in-sample...")
        in_sample_pred = fitted_model.predict(start=0, end=len(train_data)-1)

        # Previsioni out-of-sample (forecasting per il 2024)
        print("Generazione forecast out-of-sample...")
        forecast = fitted_model.forecast(steps=len(test_data))

        # Visualizzazione risultati
        plot_sarima_results(train_data, test_data, in_sample_pred, forecast)

        # Calcolo accuratezza
        accuracy = forecast_accuracy(forecast, test_data)
        print_accuracy_results(accuracy, "SARIMA")

        return in_sample_pred, forecast

    except Exception as e:
        print(f"Errore nell'addestramento del modello SARIMA: {e}")
        print("Tentativo con parametri semplificati...")

        # Fallback con parametri più semplici
        try:
            order_simple = (1, 0, 1)
            seasonal_order_simple = (1, 0, 1, 52)

            model_simple = SARIMAX(train_data,
                                 order=order_simple,
                                 seasonal_order=seasonal_order_simple,
                                 enforce_stationarity=False,
                                 enforce_invertibility=False)

            fitted_model_simple = model_simple.fit(disp=False, maxiter=50)

            in_sample_pred = fitted_model_simple.predict(start=0, end=len(train_data)-1)
            forecast = fitted_model_simple.forecast(steps=len(test_data))

            plot_sarima_results(train_data, test_data, in_sample_pred, forecast)

            accuracy = forecast_accuracy(forecast, test_data)
            print_accuracy_results(accuracy, "SARIMA (parametri semplificati)")

            return in_sample_pred, forecast

        except Exception as e2:
            print(f"Errore anche con parametri semplificati: {e2}")
            # Ritorna previsioni naive come fallback
            naive_forecast = np.full(len(test_data), np.mean(train_data))
            accuracy = forecast_accuracy(naive_forecast, test_data)
            print_accuracy_results(accuracy, "SARIMA (fallback - media)")

            return np.full(len(train_data), np.mean(train_data)), naive_forecast

def plot_sarima_results(train_data, test_data, predictions, forecast):
    """
    Visualizza i risultati del modello SARIMA

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
    plt.title('SARIMA - Previsioni In-Sample (Training Data)')
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
             label='Forecast SARIMA', linewidth=2)

    plt.axvline(x=0, color='red', linestyle=':', alpha=0.5, label='Inizio Forecast')
    plt.title('SARIMA - Forecast Out-of-Sample (2024)')
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