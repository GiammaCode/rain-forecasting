from preprocessing import (
    load_and_plot_csv,
    flatten_series,
    get_train_test,
    smooth_series,
    describe_series,
    check_stationarity
)

# Caricamento e preprocessing
df = load_and_plot_csv('../data/Pioggia_Settimanale_Emilia-Romagna2.csv')
data = flatten_series(df)
train, test = get_train_test(df)

# Smoothing + descrittive + ADF
smoothed = smooth_series(data)
describe_series(data)
check_stationarity(data)

# SARIMA
from statsModel import run_sarima

accuracy, forecast = run_sarima(train, test)
print(accuracy)

# NN
import torch
from neuralModel import create_dataset, NeuralForecaster, train_model, recursive_forecast, plot_results, forecast_accuracy

look_back = 52
trainX, trainy = create_dataset(train, look_back)
trainX = torch.FloatTensor(trainX.squeeze())
trainy = torch.FloatTensor(trainy.squeeze())

model = NeuralForecaster(input_size=look_back)
model = train_model(model, trainX, trainy)

# Predizione sui dati di training
train_pred = model(trainX).detach().numpy()
# Forecasting ricorsivo
forecast = recursive_forecast(model, trainX[-1], len(test))
# Plot dei risultati
plot_results(train, train_pred, forecast, look_back, test)
# Valutazione
print(forecast_accuracy(forecast, test))


#XGBOOST
from notNeuralModel import train_xgb_model, recursive_forecast_xgb, plot_xgb_forecast, forecast_accuracy
look_back = 52
n_forecast = len(test)

model, x_train, y_test = train_xgb_model(train, look_back, n_forecast)

x_start = x_train[-1]
forecast = recursive_forecast_xgb(model, x_start, n_forecast)

plot_xgb_forecast(test, forecast)

metrics = forecast_accuracy(forecast, test)
print(metrics)

