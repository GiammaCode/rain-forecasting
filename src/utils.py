import numpy as np

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
