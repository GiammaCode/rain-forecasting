"""
Modulo per il modello XGBoost per il forecasting delle piogge settimanali
"""

import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from preprocessing import create_dataset, forecast_accuracy
import warnings
warnings.filterwarnings('ignore')
from utils import *


def tune_xgb_model(x_train, y_train, n_iter=20):
    """
    Ottimizza gli iperparametri del modello XGBoost

    Args:
        x_train: Dati di training
        y_train: Target di training
        n_iter: Numero di iterazioni per la ricerca random

    Returns:
        XGBRegressor: Modello ottimizzato
    """
    param_dist = {
        'n_estimators': [100, 300, 500, 700, 1000],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5]
    }

    search = RandomizedSearchCV(
        XGBRegressor(objective='reg:squarederror', random_state=42),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1,
        random_state=42
    )

    print("Ottimizzazione iperparametri in corso...")
    search.fit(x_train, y_train)
    print("Best parameters:", search.best_params_)
    return search.best_estimator_


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


def xgboost_forecast(train_data, test_data, tune_hyperparams=False):
    """
    Addestra un modello XGBoost e genera previsioni

    Args:
        train_data (np.array): Dati di training (2014-2023)
        test_data (np.array): Dati di test (2024)
        tune_hyperparams (bool): Se ottimizzare gli iperparametri

    Returns:
        tuple: (predictions_in_sample, forecast_out_sample)
    """

    print("Preparazione dati per XGBoost...")

    # Parametri
    look_back = 104  # Usa 104 settimane (2 anni) per predire la settimana successiva

    # Creazione del dataset per supervised learning
    X, y = create_dataset(train_data, look_back)

    print(f"Dataset creato: X shape = {X.shape}, y shape = {y.shape}")
    print(f"Usando {look_back} settimane per predire la settimana successiva")

    try:
        print("Addestramento modello XGBoost...")

        if tune_hyperparams:
            # Ottimizzazione iperparametri
            model = tune_xgb_model(X, y, n_iter=20)
        else:
            # Parametri predefiniti
            model = XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1000,
                max_depth=5,
                learning_rate=0.01,
                subsample=0.6,
                colsample_bytree=0.8,
                gamma=1,
                random_state=42
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

    except Exception as e:
        print(f"Errore nell'addestramento del modello XGBoost: {e}")
        return None, None


