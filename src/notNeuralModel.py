import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


def create_dataset(dataset, look_back):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:i + look_back])
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)


def train_xgb_model(train, look_back, n_forecast):
    X, y = create_dataset(train, look_back)
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
    plt.plot(forecast, "-o", label="Forecast")
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