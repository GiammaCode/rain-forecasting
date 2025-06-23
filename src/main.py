from preprocessing import (
    load_and_plot_csv,
    flatten_series,
    get_train_test,
    smooth_series,
    describe_series,
    check_stationarity
)
from utils import *

# Caricamento e preprocessing
df = load_and_plot_csv('../data/Pioggia_Settimanale_Emilia-Romagna.csv')
data = flatten_series(df)
train, test = get_train_test(df)

# Smoothing + descrittive + ADF
smoothed = smooth_series(data)
smoothedTrain = smooth_series(train)
describe_series(data)
check_stationarity(data)

# SARIMA
from statsModel import run_sarima
forecastStats = run_sarima(train, test)
print("Sarimax Accuracy:")
print(forecast_accuracy(forecastStats, test))


# NN
import torch
from neuralModel import NeuralForecaster, train_model, recursive_forecast, plot_results
from neuralModel import set_seed

look_back = 104
trainX, trainy = create_dataset(train, look_back)
trainX = torch.FloatTensor(trainX.squeeze())
trainy = torch.FloatTensor(trainy.squeeze())

set_seed(42)
model = NeuralForecaster()
model = train_model(model, trainX, trainy)
# Predizione sui dati di training
train_pred = model(trainX).detach().numpy()
# Forecasting ricorsivo
forecastNN = recursive_forecast(model, trainX[-1], len(test))
# Plot dei risultati
plot_results(forecastNN, test)
# Valutazione
print("NN Accuracy: ")
#print(forecast_accuracy(forecastNN, test))


#XGBOOST
from notNeuralModel import *

look_back = 104
n_forecast = len(test)
X, y = create_dataset(train, look_back)

model, x_train, y_test = train_xgb_model(X, y, n_forecast)

x_start = x_train[-1]
forecastXGB = recursive_forecast_xgb(model, x_start, n_forecast)

plot_xgb_forecast(test, forecastXGB)

print("XGBoost Accuracy: ")
print(forecast_accuracy(forecastXGB, test))

plot_forecast_comparison(test, forecastStats, forecastNN, forecastXGB)

