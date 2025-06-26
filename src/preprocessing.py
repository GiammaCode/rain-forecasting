"""
Modulo di preprocessing per i dati delle piogge settimanali
Gestisce il caricamento, la pulizia e la preparazione dei dati per il forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def load_and_preprocess_data(filepath):
    """
    Carica e preprocessa i dati delle piogge settimanali

    Args:
        filepath (str): Percorso del file CSV

    Returns:
        tuple: (train_data, test_data, full_data) - array numpy dei dati preprocessati
    """

    print("Caricamento dati dal file CSV...")
    df = pd.read_csv(filepath)

    # Conversione del dataset in formato serie temporale
    # Concateniamo tutti gli anni dal 2014 al 2023 per il training
    train_data = []
    for year in range(2014, 2024):  # 2014-2023 per training
        if str(year) in df.columns:
            train_data.extend(df[str(year)].values)

    # Dati 2024 per il test
    test_data = df['2024'].values

    # Dataset completo
    full_data = []
    for year in range(2014, 2025):  # 2014-2024
        if str(year) in df.columns:
            full_data.extend(df[str(year)].values)

    # Conversione in array numpy
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    full_data = np.array(full_data)

    print(f"Dati caricati: {len(train_data)} settimane training, {len(test_data)} settimane test")

    # Analisi esplorativa dei dati
    print("\n--- ANALISI ESPLORATIVA DEI DATI ---")
    analyze_data(train_data, "Training Data (2014-2023)")
    analyze_data(test_data, "Test Data (2024)")

    # Gestione degli outliers e valori anomali
    print("\n--- GESTIONE OUTLIERS ---")
    train_data_clean = handle_outliers(train_data)
    test_data_clean = handle_outliers(test_data)
    full_data_clean = np.concatenate([train_data_clean, test_data_clean])

    # Test di stazionarietà
    print("\n--- TEST DI STAZIONARIETÀ ---")
    check_stationarity(train_data_clean)

    # Visualizzazione delle funzioni di autocorrelazione
    visualize_autocorrelation(train_data_clean)

    return train_data_clean, test_data_clean, full_data_clean

def analyze_data(data, label):
    """
    Analizza statisticamente una serie di dati

    Args:
        data (np.array): Serie di dati da analizzare
        label (str): Etichetta per l'output
    """

    print(f"\n{label}:")
    print(f"  Lunghezza: {len(data)}")
    print(f"  Min: {np.min(data):.2f} mm")
    print(f"  Max: {np.max(data):.2f} mm")
    print(f"  Media: {np.mean(data):.2f} mm")
    print(f"  Mediana: {np.median(data):.2f} mm")
    print(f"  Deviazione Standard: {np.std(data):.2f} mm")
    print(f"  Valori zero: {np.sum(data == 0)} ({np.sum(data == 0)/len(data)*100:.1f}%)")

    # Percentili
    print(f"  25° percentile: {np.percentile(data, 25):.2f} mm")
    print(f"  75° percentile: {np.percentile(data, 75):.2f} mm")

    # Visualizzazione distribuzione
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(data)
    plt.title(f'{label} - Serie Temporale')
    plt.xlabel('Settimane')
    plt.ylabel('Pioggia (mm)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
    plt.title(f'{label} - Distribuzione')
    plt.xlabel('Pioggia (mm)')
    plt.ylabel('Frequenza')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.boxplot(data)
    plt.title(f'{label} - Box Plot')
    plt.ylabel('Pioggia (mm)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def handle_outliers(data, method='iqr', threshold=3):
    """
    Gestisce gli outliers nei dati

    Args:
        data (np.array): Serie di dati
        method (str): Metodo per identificare outliers ('iqr' o 'zscore')
        threshold (float): Soglia per z-score

    Returns:
        np.array: Dati con outliers gestiti
    """

    data_clean = data.copy()

    if method == 'iqr':
        # Metodo Interquartile Range
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_mask = (data < lower_bound) | (data > upper_bound)
        outliers_count = np.sum(outliers_mask)

        print(f"Outliers identificati con metodo IQR: {outliers_count}")
        print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")

        # Non rimuoviamo gli outliers ma li segnaliamo
        # Per i dati di pioggia, valori molto alti potrebbero essere eventi meteorologici estremi reali
        if outliers_count > 0:
            print(f"Valori outlier: {data[outliers_mask]}")

    elif method == 'zscore':
        # Metodo Z-Score
        z_scores = np.abs(stats.zscore(data))
        outliers_mask = z_scores > threshold
        outliers_count = np.sum(outliers_mask)

        print(f"Outliers identificati con Z-score (threshold={threshold}): {outliers_count}")

        if outliers_count > 0:
            print(f"Valori outlier: {data[outliers_mask]}")

    # Gestione dei valori zero
    zero_count = np.sum(data == 0)
    zero_percentage = zero_count / len(data) * 100

    print(f"Valori zero: {zero_count} ({zero_percentage:.1f}%)")
    print("I valori zero vengono mantenuti (rappresentano settimane senza pioggia)")

    # Per i dati meteorologici, manteniamo tutti i valori inclusi gli outliers
    # Gli eventi estremi sono parte della variabilità naturale

    return data_clean

def check_stationarity(data):
    """
    Verifica la stazionarietà della serie temporale usando il test ADF

    Args:
        data (np.array): Serie temporale da testare
    """

    print("Test di Augmented Dickey-Fuller per la stazionarietà:")

    # Test ADF
    adf_result = adfuller(data)

    print(f"ADF Statistic: {adf_result[0]:.6f}")
    print(f"p-value: {adf_result[1]:.6f}")
    print("Valori critici:")
    for key, value in adf_result[4].items():
        print(f"\t{key}: {value:.3f}")

    # Interpretazione
    if adf_result[1] <= 0.05:
        print("Risultato: Rifiutiamo l'ipotesi nulla. La serie è STAZIONARIA.")
    else:
        print("Risultato: Non possiamo rifiutare l'ipotesi nulla. La serie NON è stazionaria.")
        print("Potrebbe essere necessaria una trasformazione (differenziazione).")

def visualize_autocorrelation(data, lags=52):
    """
    Visualizza le funzioni di autocorrelazione (ACF) e autocorrelazione parziale (PACF)

    Args:
        data (np.array): Serie temporale
        lags (int): Numero di lag da visualizzare
    """

    print(f"Visualizzazione ACF e PACF con {lags} lags...")

    plt.figure(figsize=(15, 6))

    # Autocorrelation Function
    plt.subplot(1, 2, 1)
    plot_acf(data, lags=lags, ax=plt.gca())
    plt.title('Autocorrelation Function (ACF)')
    plt.grid(True, alpha=0.3)

    # Partial Autocorrelation Function
    plt.subplot(1, 2, 2)
    plot_pacf(data, lags=lags, ax=plt.gca())
    plt.title('Partial Autocorrelation Function (PACF)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Analisi della stagionalità
    print("\nAnalisi stagionalità (settimanale - 52 settimane):")
    seasonal_acf = []
    for lag in [52, 104, 156]:  # 1, 2, 3 anni
        if lag < len(data):
            correlation = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            seasonal_acf.append(correlation)
            print(f"Lag {lag} settimane ({lag//52} anni): correlazione = {correlation:.3f}")

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

# Funzione di accuratezza (copiata dall'esempio originale)
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