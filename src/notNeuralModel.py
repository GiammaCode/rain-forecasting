import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

def tune_xgb_model(x_train, y_train, n_iter=20):
    param_dist = {
        'n_estimators': [100, 300, 500, 700, 1000],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5]
    }

    search = RandomizedSearchCV(
        XGBRegressor(objective='reg:squarederror'),
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1,
        random_state=42
    )
    search.fit(x_train, y_train)
    print("Best parameters:", search.best_params_)
    return search.best_estimator_

def train_xgb_model(X, y,  n_forecast):
    x_train, _ = X[:-n_forecast], X[-n_forecast:]
    y_train, y_test = y[:-n_forecast], y[-n_forecast:]

    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.6,
        colsample_bytree=0.8,
        gamma=1
    )

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
