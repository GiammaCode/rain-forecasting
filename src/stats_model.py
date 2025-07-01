"""
Modulo per il modello SARIMA per il forecasting delle piogge settimanali
"""

from utils import *
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    print("Parametri: order=(1,1,2), seasonal_order=(1,1,1,52)")

    # Definizione parametri del modello
    # order = (p, d, q) - parametri non stagionali
    # seasonal_order = (P, D, Q, s) - parametri stagionali
    order = (1, 1, 2)
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
        plot_results(train_data, test_data, in_sample_pred, forecast, "SARIMA")

        # Calcolo accuratezza
        accuracy = forecast_accuracy(forecast, test_data)
        print_accuracy_results(accuracy, "SARIMA")

        return in_sample_pred, forecast

    except Exception as e:
        print(f"Errore nell'addestramento del modello SARIMA: {e}")

