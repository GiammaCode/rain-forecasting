import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


def train_xgb_model(X, y,  n_forecast):
    x_train, _ = X[:-n_forecast], X[-n_forecast:]
    y_train, y_test = y[:-n_forecast], y[-n_forecast:]

    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(x_train, y_train)

    return model, x_train, y_test


def recursive_forecast_xgb(model, x_start, n_forecast):
    xinput = x_start.copy()
    forecast = []
    for _ in range(n_forecast):
        pred = model.predict(xinput.reshape(1, -1))[0]
        forecast.append(pred)
        xinput = np.roll(xinput, -1)
        xinput[-1] = pred
    return forecast


def plot_xgb_forecast(test, forecast):
    plt.figure()
    plt.plot(test, label="Actual")
    plt.plot(forecast, label="Forecast")
    plt.legend()
    plt.grid()
    plt.title("XGBoost Forecast")
    plt.show()


def forecast_accuracy(forecast, actual):
    forecast = np.array(forecast)
    actual = np.array(actual)
    mae = np.mean(np.abs(forecast - actual))
    mse = np.mean((forecast - actual) ** 2)
    rmse = np.sqrt(mse)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}