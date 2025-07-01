import numpy as np
import matplotlib.pyplot as plt
from preprocessing import load_and_preprocess_data
from stats_model import sarima_forecast
from NN_model import neural_network_forecast
from XGBoost_model import xgboost_forecast
from dm_test import dm_test

# Configurazione per la riproducibilità
import random
import torch

np.random.seed(123)
random.seed(123)
torch.manual_seed(123)


def main():
    print("=" * 80)
    print("PROGETTO FORECASTING PIOGGE SETTIMANALI - EMILIA-ROMAGNA")
    print("=" * 80)

    # 1. Caricamento e preprocessing dei dati
    print("\n1. CARICAMENTO E PREPROCESSING DEI DATI")
    print("-" * 50)

    train_data, test_data, full_data = load_and_preprocess_data('../data/Pioggia_Settimanale_Emilia-Romagna.csv')

    print(f"Dati di training: {len(train_data)} settimane (2014-2023)")
    print(f"Dati di test: {len(test_data)} settimane (2024)")
    print(f"Dataset completo: {len(full_data)} settimane")

    # Visualizzazione della serie temporale completa
    plt.figure(figsize=(15, 6))
    plt.plot(range(len(full_data)), full_data, label='Serie completa', alpha=0.7)
    plt.axvline(x=len(train_data), color='red', linestyle='--', label='Train/Test split')
    plt.title('Serie Temporale Piogge Settimanali Emilia-Romagna (2014-2024)')
    plt.xlabel('Settimane')
    plt.ylabel('Pioggia (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. Modello SARIMA
    print("\n2. MODELLO SARIMA")
    print("-" * 50)

    sarima_pred, sarima_forecast_vals = sarima_forecast(train_data, test_data)

    # 3. Rete Neurale
    print("\n3. RETE NEURALE")
    print("-" * 50)

    nn_pred, nn_forecast_vals = neural_network_forecast(train_data, test_data)

    # 4. XGBoost
    print("\n4. XGBOOST")
    print("-" * 50)

    xgb_pred, xgb_forecast_vals = xgboost_forecast(train_data, test_data)

    # Plot di confronto
    plt.figure(figsize=(15, 8))

    # Plot dei dati di training (ultime 52 settimane per contesto)
    plt.plot(range(-52, 0), train_data[-52:], 'k-', label='Training (2023)', alpha=0.7)

    # Plot dei dati reali 2024 e previsioni
    plt.plot(range(len(test_data)), test_data, 'ko-', label='Dati Reali 2024', markersize=4)
    plt.plot(range(len(sarima_forecast_vals)), sarima_forecast_vals, '--',
              label='SARIMA', linewidth=2)
    plt.plot(range(len(nn_forecast_vals)), nn_forecast_vals, ':',
              label='Neural Network', linewidth=2)
    plt.plot(range(len(xgb_forecast_vals)), xgb_forecast_vals, '-.',
              label='XGBoost', linewidth=2)

    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Inizio Forecast')
    plt.title('Confronto Modelli di Forecasting - Piogge 2024')
    plt.xlabel('Settimane dal 2024')
    plt.ylabel('Pioggia (mm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 6. Test statistici di confronto (Diebold-Mariano)
    print("\n6. TEST STATISTICI DI CONFRONTO")
    print("-" * 50)


    # Confronto SARIMA vs Neural Network
    dm_sarima_nn = dm_test(test_data, nn_forecast_vals, sarima_forecast_vals, h=1, crit="MSE")
    print(f"SARIMA vs Neural Network - DM stat: {dm_sarima_nn.DM:.4f}, p-value: {dm_sarima_nn.p_value:.4f}")
    # Confronto SARIMA vs XGBoost
    dm_sarima_xgb = dm_test(test_data, xgb_forecast_vals, sarima_forecast_vals, h=1, crit="MSE")
    print(f"SARIMA vs XGBoost - DM stat: {dm_sarima_xgb.DM:.4f}, p-value: {dm_sarima_xgb.p_value:.4f}")
    # Confronto Neural Network vs XGBoost
    dm_nn_xgb = dm_test(test_data, xgb_forecast_vals, nn_forecast_vals, h=1, crit="MSE")
    print(f"Neural Network vs XGBoost - DM stat: {dm_nn_xgb.DM:.4f}, p-value: {dm_nn_xgb.p_value:.4f}")
    print("\nInterpretazione:")
    print("- Se |DM stat| > 1.96 e p-value < 0.05: differenza significativa tra i modelli")
    print("- Se |DM stat| <= 1.96 o p-value >= 0.05: nessuna differenza significativa")



if __name__ == "__main__":
    main()