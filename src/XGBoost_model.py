"""
Modulo per il modello XGBoost per il forecasting delle piogge settimanali
"""

import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from preprocessing import create_dataset, forecast_accuracy
import warnings
warnings.filterwarnings('ignore')

def xgboost_forecast(train_data, test_data):
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
    look_back = 52  # Usa 52 settimane (1 anno) per predire la settimana successiva

    # Creazione del dataset per supervised learning
    X, y = create_dataset(train_data, look_back)

    print(f"Dataset creato: X shape = {X.shape}, y shape = {y.shape}")
    print(f"Usando {look_back} settimane per predire la settimana successiva")

    # Preparazione dei dati per il training
    # Non usiamo le ultime n settimane per validation, ma tutto per training
    X_train = X
    y_train = y

    print("Inizializzazione e addestramento del modello XGBoost...")

    # Configurazione del modello XGBoost
    # Parametri ottimizzati per series temporali
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,          # Numero di alberi
        max_depth=6,                # Profondità massima degli alberi
        learning_rate=0.1,          # Learning rate
        subsample=0.8,              # Frazione di samples per albero
        colsample_bytree=0.8,       # Frazione di features per albero
        random_state=123,           # Per riproducibilità
        n_jobs=-1,                  # Usa tutti i processori
        early_stopping_rounds=50,   # Early stopping
        eval_metric='rmse'          # Metrica di valutazione
    )

    print("Parametri del modello:")
    print(f"  - N. stimatori: {model.n_estimators}")
    print(f"  - Max depth: {model.max_depth}")
    print(f"  - Learning rate: {model.learning_rate}")

    # Training del modello
    print("Inizio training...")

    # Per early stopping, dividiamo i dati in train e validation
    split_index = int(0.8 * len(X_train))
    X_train_split = X_train[:split_index]
    y_train_split = y_train[:split_index]
    X_val_split = X_train[split_index:]
    y_val_split = y_train[split_index:]

    # Fit con validation set per early stopping
    model.fit(
        X_train_split, y_train_split,
        eval_set=[(X_val_split, y_val_split)],
        verbose=100  # Stampa ogni 100 iterazioni
    )

    print(f"Training completato! Migliore iterazione: {model.best_iteration}")

    # Re-fit su tutti i dati di training con il numero ottimale di iterazioni
    model_final = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=model.best_iteration,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=123,
        n_jobs=-1
    )

    model_final.fit(X_train, y_train)

    # Previsioni in-sample
    print("Generazione previsioni in-sample...")
    train_predictions = model_final.predict(X_train)

    # Feature importance
    feature_importance = model_final.feature_importances_
    plot_feature_importance(feature_importance, look_back)

    # Forecasting ricorsivo per i dati di test
    print("Generazione forecast out-of-sample con approccio ricorsivo...")

    # Inizializza con l'ultima finestra di training
    input_sequence = train_data[-look_back:].copy()
    forecast_values = []

    for step in range(len(test_data)):
        # Usa la sequenza corrente per fare previsione
        prediction = model_final.predict(input_sequence.reshape(1, -1))[0]
        forecast_values.append(prediction)

        # Aggiorna la sequenza: shift a sinistra e aggiungi la previsione
        input_sequence = np.roll(input_sequence, -1)
        input_sequence[-1] = prediction

        if (step + 1) % 10 == 0:
            print(f"Generata previsione per settimana {step + 1}/52")

    forecast_values = np.array(forecast_values)

    # Visualizzazione risultati
    plot_xgboost_results(train_data, test_data, train_predictions, forecast_values, look_back)

    # Calcolo accuratezza
    accuracy = forecast_accuracy(forecast_values, test_data)
    print_xgboost_accuracy_results(accuracy)

    return train_predictions, forecast_values

def plot_feature_importance(feature_importance, look_back):
    """
    Visualizza l'importanza delle features (lag weeks) nel modello XGBoost

    Args:
        feature_importance (np.array): Importanza delle features
        look_back (int): Numero di lag utilizzati
    """

    plt.figure(figsize=(15, 6))

    # Plot 1: Feature importance completa
    plt.subplot(1, 2, 1)
    weeks_back = range(1, look_back + 1)
    plt.bar(weeks_back, feature_importance, alpha=0.7)
    plt.title('Feature Importance - Tutte le settimane')
    plt.xlabel('Settimane precedenti')
    plt.ylabel('Importanza')
    plt.grid(True, alpha=0.3)

    # Plot 2: Top 10 features più importanti
    plt.subplot(1, 2, 2)
    top_indices = np.argsort(feature_importance)[-10:]
    top_importance = feature_importance[top_indices]
    top_weeks = np.array(weeks_back)[top_indices]

    plt.barh(range(10), top_importance, alpha=0.7)
    plt.yticks(range(10), [f'Settimana -{w}' for w in top_weeks])
    plt.title('Top 10 Features più Importanti')
    plt.xlabel('Importanza')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Stampa le top features
    print("\nTop 10 settimane più importanti per le previsioni:")
    for i, (week, importance) in enumerate(zip(top_weeks, top_importance)):
        print(f"{i+1:2d}. Settimana -{week:2d}: {importance:.4f}")

def plot_xgboost_results(train_data, test_data, predictions, forecast, look_back):
    """
    Visualizza i risultati del modello XGBoost

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
    plt.title('XGBoost - Previsioni In-Sample (Training Data)')
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
    plt.plot(range(len(forecast)), forecast, 'b--',
             label='Forecast XGBoost', linewidth=2)

    plt.axvline(x=0, color='red', linestyle=':', alpha=0.5, label='Inizio Forecast')
    plt.title('XGBoost - Forecast Out-of-Sample (2024)')
    plt.xlabel('Settimane dal 2024')
    plt.ylabel('Pioggia (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def print_xgboost_accuracy_results(accuracy):
    """
    Stampa le metriche di accuratezza per XGBoost

    Args:
        accuracy (dict): Dizionario con le metriche
    """

    print(f"\n--- RISULTATI ACCURATEZZA XGBOOST ---")
    print(f"MAPE (Mean Absolute Percentage Error): {accuracy['mape']:.4f} ({accuracy['mape']*100:.2f}%)")
    print(f"ME (Mean Error): {accuracy['me']:.4f}")
    print(f"MAE (Mean Absolute Error): {accuracy['mae']:.4f}")
    print(f"MPE (Mean Percentage Error): {accuracy['mpe']:.4f} ({accuracy['mpe']*100:.2f}%)")
    print(f"RMSE (Root Mean Square Error): {accuracy['rmse']:.4f}")
    print(f"Correlazione: {accuracy['corr']:.4f}")
    print("-" * 50)

# Funzione di utilità per analisi aggiuntive
def analyze_residuals(actual, predicted, model_name="XGBoost"):
    """
    Analizza i residui delle previsioni

    Args:
        actual (np.array): Valori reali
        predicted (np.array): Valori predetti
        model_name (str): Nome del modello
    """

    residuals = actual - predicted

    plt.figure(figsize=(15, 5))

    # Plot 1: Residui nel tempo
    plt.subplot(1, 3, 1)
    plt.plot(residuals, 'o-', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.title(f'{model_name} - Residui nel Tempo')
    plt.xlabel('Settimane')
    plt.ylabel('Residui (mm)')
    plt.grid(True, alpha=0.3)

    # Plot 2: Distribuzione dei residui
    plt.subplot(1, 3, 2)
    plt.hist(residuals, bins=15, alpha=0.7, edgecolor='black')
    plt.title(f'{model_name} - Distribuzione Residui')
    plt.xlabel('Residui (mm)')
    plt.ylabel('Frequenza')
    plt.grid(True, alpha=0.3)

    # Plot 3: Scatter plot predetto vs reale
    plt.subplot(1, 3, 3)
    plt.scatter(predicted, actual, alpha=0.7)
    min_val = min(np.min(actual), np.min(predicted))
    max_val = max(np.max(actual), np.max(predicted))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
    plt.xlabel('Valori Predetti')
    plt.ylabel('Valori Reali')
    plt.title(f'{model_name} - Predetto vs Reale')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Statistiche dei residui
    print(f"\n--- ANALISI RESIDUI {model_name.upper()} ---")
    print(f"Media residui: {np.mean(residuals):.4f}")
    print(f"Std residui: {np.std(residuals):.4f}")
    print(f"Min residuo: {np.min(residuals):.4f}")
    print(f"Max residuo: {np.max(residuals):.4f}")