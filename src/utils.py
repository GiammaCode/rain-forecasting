import numpy as np
import matplotlib.pyplot as plt


# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.where(actual == 0, 1, np.abs(actual)))
    me = np.mean(forecast - actual) # ME
    mae = np.mean(np.abs(forecast - actual)) # MAE
    mpe = np.mean((forecast - actual) / np.where(actual == 0, 1, np.abs(actual)))
    rmse = np.mean((forecast - actual)**2)**.5 # RMSE
    corr = np.corrcoef(forecast, actual)[0,1] # correlation coeff
    return({'mape':mape, 'me':me, 'mae': mae, 'mpe': mpe, 'rmse':rmse, 'corr':corr})

# Converts an array of values into two np arrays
def create_dataset(arrdata, look_back=1):
    dataX, dataY = [], []
    for i in range(len(arrdata) - look_back):
        a = arrdata[i:(i + look_back)]
        dataX.append(a)
        dataY.append(arrdata[i + look_back])
    return np.array(dataX), np.array(dataY)

def plot_forecast_comparison(actual, forecast_stat, forecast_nn, forecast_tree):
    weeks = list(range(1, len(actual) + 1))

    plt.figure(figsize=(12, 6))
    plt.plot(weeks, actual, label="Dati Reali", color="blue", linewidth=2)
    plt.plot(weeks, forecast_stat, label="Forecast Statistico (SARIMA)", color="red")
    plt.plot(weeks, forecast_nn, label="Forecast Rete Neurale", color="yellow")
    plt.plot(weeks, forecast_tree, label="Forecast Decision Tree", color="green")

    plt.title("Confronto Previsioni 2024")
    plt.xlabel("Settimana")
    plt.ylabel("Pioggia (mm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
