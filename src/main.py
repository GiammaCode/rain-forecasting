from preprocessing import (
    load_and_plot_csv,
    flatten_series,
    get_train_test,
    smooth_series,
    describe_series,
    check_stationarity
)
from statsModel import run_sarima

# Caricamento e preprocessing
df = load_and_plot_csv('../data/Pioggia_Settimanale_Emilia-Romagna2.csv')
data = flatten_series(df)
train, test = get_train_test(df)

# Smoothing + descrittive + ADF
smoothed = smooth_series(data)
describe_series(data)
check_stationarity(data)

# SARIMA
accuracy, forecast = run_sarima(train, test)
print(accuracy)




