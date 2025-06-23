import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX


def run_sarima(train, test, order=(0, 0, 0), seasonal_order=(1, 1, 0, 52)):
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    fit = model.fit()
    pred_in_sample = fit.predict(0, len(train))
    forecast = fit.forecast(steps=len(test))

    # Plot forecast
    plt.plot(test, label="actual")
    plt.plot(forecast,  label="forecast")
    plt.legend()
    plt.grid()
    plt.title("Statistical model Forecast - 2024")
    plt.xlabel("Settimana del 2024")
    plt.ylabel("Pioggia (mm)")
    plt.show()

    return forecast
