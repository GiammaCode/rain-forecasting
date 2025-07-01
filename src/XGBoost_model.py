"""
Modulo per il modello XGBoost per il forecasting delle piogge settimanali
"""

from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')
from utils import *


def recursive_forecast_xgb(model, x_start, n_forecast):
    """
    Genera forecast ricorsivo con XGBoost

    Args:
        model: Modello XGBoost addestrato
        x_start: Sequenza iniziale per il forecast
        n_forecast: Numero di step da prevedere

    Returns:
        list: Previsioni
    """
    xinput = x_start.copy()
    forecast = []

    for step in range(n_forecast):
        pred = model.predict(xinput.reshape(1, -1))[0]
        forecast.append(pred)
        xinput = np.roll(xinput, -1)
        xinput[-1] = pred

        if (step + 1) % 10 == 0:
            print(f"Generata previsione per settimana {step + 1}/{n_forecast}")

    return forecast


def xgboost_forecast(train_data, test_data, tune_hyperparams= True):
    """
    Addestra un modello XGBoost e genera previsioni

    Args:
        train_data (np.array): Dati di training (2014-2023)
        test_data (np.array): Dati di test (2024)

    Returns:
        tuple: (predictions_in_sample, forecast_out_sample)
    """

    print("Preparazione dati per XGBoost...")

    # Parametri
    look_back = 52  # Usa 104 settimane (2 anni) per predire la settimana successiva

    # Creazione del dataset per supervised learning
    X, y = create_dataset(train_data, look_back)

    print(f"Dataset creato: X shape = {X.shape}, y shape = {y.shape}")
    print(f"Usando {look_back} settimane per predire la settimana successiva")

    print("Addestramento modello XGBoost...")

    # Parametri predefiniti
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.8,
        gamma=1,
        random_state=60
        )
    print("Parametri utilizzati:")
    print(f"n_estimators: 1000, max_depth: 5, learning_rate: 0.01")
    print(f"subsample: 0.6, colsample_bytree: 0.8, gamma: 1")

    # Addestramento del modello
    print("Fitting del modello in corso...")
    model.fit(X, y)
    print("Modello addestrato con successo!")

    # Previsioni in-sample (sui dati di training)
    print("Generazione previsioni in-sample...")
    in_sample_pred = model.predict(X)

    # Previsioni out-of-sample (forecasting ricorsivo per il 2024)
    print("Generazione forecast out-of-sample con approccio ricorsivo...")

    # Inizializza con l'ultima finestra di training
    x_start = train_data[-look_back:]
    forecast = recursive_forecast_xgb(model, x_start, len(test_data))
    forecast = np.array(forecast)

    # Visualizzazione risultati
    plot_results(train_data, test_data, in_sample_pred, forecast, "XGBoost", look_back)

    # Calcolo accuratezza
    accuracy = forecast_accuracy(forecast, test_data)
    print_accuracy_results(accuracy, "XGBoost")

    return in_sample_pred, forecast


