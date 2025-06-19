import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.where(actual == 0, 1, np.abs(actual)))
    me = np.mean(forecast - actual)
    mae = np.mean(np.abs(forecast - actual))
    mpe = np.mean((forecast - actual) / np.where(actual == 0, 1, np.abs(actual)))
    rmse = np.mean((forecast - actual) ** 2) ** .5
    corr = np.corrcoef(forecast, actual)[0, 1]
    return {'mape': mape, 'me': me, 'mae': mae, 'mpe': mpe, 'rmse': rmse, 'corr': corr}


def run_sarima(train, test, order=(0, 0, 0), seasonal_order=(1, 1, 0, 52)):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    fit = model.fit()
    pred_in_sample = fit.predict(0, len(train))
    forecast = fit.forecast(steps=len(test))

    # Plot training fit
    plt.plot(pred_in_sample, label="predict")
    plt.plot(train, label="actual")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot forecast
    plt.plot(forecast, label="forecast")
    plt.plot(test, "-o", label="actual")
    plt.legend()
    plt.grid()
    plt.title("Sarima")
    plt.show()

    return forecast_accuracy(forecast, test), forecast
