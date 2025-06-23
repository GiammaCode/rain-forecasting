# Rain Forecasting - Emilia-Romagna

Previsione settimanale delle piogge nella regione Emilia-Romagna, basata su modelli statistici (SARIMA, SARIMAX) e di machine learning (Neural Network, XGBoost).  
Il progetto utilizza dati meteorologici reali ed è pensato per essere facilmente esteso con nuove feature.

---

## Struttura del progetto
```
rain-forecasting/
├── data/ # Dataset CSV e immagini di output
│ ├── Pioggia_Settimanale_Emilia-Romagna.csv
│ └── ...
├── docs/ # Documentazione per GitHub Pages
│ ├── forecasting_neurale.md
│ ├── forecasting_statistico.md
│ └── img/
├── src/ # Codice Python
│ ├── main.py # Entry point del progetto
│ ├── neuralModel.py # Rete neurale per il forecasting
│ ├── notNeuralModel.py # Modello XGBoost
│ ├── statsModel.py # Modelli SARIMA / SARIMAX
│ ├── preprocessing.py # Caricamento e trasformazioni dati
│ └── utils.py # Funzioni di utilità
└── requirements.txt # Librerie Python richieste
```


---

## Modelli implementati

### SARIMA / SARIMAX
- Basati su `statsmodels`
- Supportano variabili esogene come `sin_week`, `cos_week`, umidità

### Neural Network
- Rete fully connected (feedforward) con `look_back = 52`
- Allenamento con MSE loss, forecasting ricorsivo

### XGBoost
- Regressore con finestre temporali
- Feature ingegnerizzate da lag

---

## Come eseguire

```bash
# Clona il progetto
git clone https://github.com/giammaCode/rain-forecasting.git
cd rain-forecasting

# Crea un ambiente virtuale
python -m venv .venv
source .venv/bin/activate  # su Windows: .venv\Scripts\activate

# Installa le dipendenze
pip install -r requirements.txt

# Esegui il progetto
python src/main.py
```


## Risultati
Sono disponibili grafici e metriche per ogni modello in:

- docs/img/
- docs/forecasting_statistico.md
- docs/forecasting_neurale.md

Esempi di metriche:
MAPE, MAE, RMSE, CORR per valutare la precisione del forecast

