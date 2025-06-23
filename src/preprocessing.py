import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf


def load_and_plot_csv(path):
    df = pd.read_csv(path)
    plt.figure(figsize=(12,6))
    for col in df.columns[:]:
        plt.plot(df.index, df[col], label=col)
    plt.title("Pioggia Settimanale per Anno - Emilia-Romagna")
    plt.xlabel("Settimana (1-52)")
    plt.ylabel("Pioggia [mm]")
    plt.legend(ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.show()
    return df

def flatten_series(df):
    data = []
    for i in range(len(df.columns)):
        data.extend(df.iloc[:, i].to_numpy())
    return data

def get_train_test(df):
    dftrain = df.iloc[:, :-1]
    train = []
    for i in range(dftrain.shape[1]):
        train.extend(dftrain.iloc[:, i].to_numpy())
    test = df.iloc[:, -1].to_numpy()
    return np.array(train), np.array(test)

def smooth_series(data, window=3):
    return pd.Series(data).rolling(window=window).mean()

def describe_series(data):
    print(f'Min: {min(data)} mm')
    print(f'Max: {max(data)} mm')
    print(f'Mean: {np.mean(data):.2f} mm')
    print(f'Median: {np.median(data):.2f} mm')
    print(f'Standard Deviation: {np.std(data):.2f} mm')

def check_stationarity(data):
    plot_acf(np.array(data))
    plt.title("Autocorrelazione - Pioggia Settimanale")
    plt.grid()
    plt.show()

    adf_result = adfuller(data)
    print(f'ADF stats: {adf_result[0]}, P-value: {adf_result[1]}')
    if adf_result[1] < 0.05:
        print("La serie è stazionaria (p-value < 0.05)")
    else:
        print("La serie NON è stazionaria (p-value >= 0.05)")
