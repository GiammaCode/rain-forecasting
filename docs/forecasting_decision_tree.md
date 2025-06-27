---
layout: default
title: XGBoost
nav_order: 5
---

# Modello XGBoost 

## Panoramica

Il modulo `XGBoost_model.py` implementa un modello XGBoost (Extreme Gradient Boosting) per il forecasting delle piogge settimanali dell'Emilia-Romagna. Questo documento descrive l'implementazione del modello, i parametri utilizzati, i risultati ottenuti e l'interpretazione delle performance.

## 1. Introduzione al Modello XGBoost

### 1.1 Definizione
XGBoost (Extreme Gradient Boosting) è un algoritmo di machine learning basato su gradient boosting che combina multiple weak learners (tipicamente decision trees) per creare un modello predittivo robusto e accurato. È particolarmente efficace per problemi di regressione e classificazione con dati strutturati.

### 1.2 Principi Fondamentali
- **Ensemble Method**: Combina previsioni di multipli decision trees
- **Gradient Boosting**: Ogni albero corregge gli errori del precedente
- **Regularization**: Controllo integrato dell'overfitting
- **Parallel Processing**: Ottimizzazione computazionale avanzata

## 2. Architettura del Modello

### 2.1 Configurazione Utilizzata
```python
XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.01,
    subsample=0.6,
    colsample_bytree=0.8,
    gamma=1,
    random_state=42
)
```

### 2.2 Parametri del Modello

#### Parametri di Base:
- **objective**: 'reg:squarederror' (regressione con errore quadratico)
- **n_estimators**: 1000 (numero di alberi nell'ensemble)
- **random_state**: 42 (per riproducibilità)

#### Parametri di Struttura:
- **max_depth**: 5 (profondità massima degli alberi)
- **learning_rate**: 0.01 (tasso di apprendimento, eta)

#### Parametri di Regularization:
- **subsample**: 0.6 (frazione di campioni per ogni albero)
- **colsample_bytree**: 0.8 (frazione di features per ogni albero)
- **gamma**: 1 (minima riduzione di loss per split)

## 3. Preparazione Dati e Features

### 3.1 Configurazione del Dataset
```python
look_back = 104  # Usa 104 settimane (2 anni) per predire la settimana successiva
```

#### Struttura dei Dati:
- **Input Features (X)**: 104 valori di pioggia settimanale precedenti
- **Target (y)**: Valore di pioggia della settimana successiva
- **Dataset Shape**: X (416, 104), y (416,)
- **Approccio**: Supervised learning con sliding window

### 3.2 Feature Engineering
Il modello utilizza un approccio di **autoregressive features**:
- Ogni riga contiene 104 settimane consecutive
- Il target è la settimana immediatamente successiva
- Creazione di 416 esempi di training dal dataset originale

## 4. Ottimizzazione Iperparametri

### 4.1 Spazio di Ricerca
Il modello include una funzione per l'ottimizzazione automatica:

```python
param_dist = {
    'n_estimators': [100, 300, 500, 700, 1000],
    'max_depth': [3, 4, 5, 6, 7, 8],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 1, 5]
}
```

### 4.2 Strategia di Ottimizzazione
- **Metodo**: RandomizedSearchCV
- **Cross-Validation**: 3-fold CV
- **Scoring**: Negative Mean Squared Error
- **Parallel Processing**: n_jobs=-1

## 5. Forecasting Ricorsivo

### 5.1 Metodologia Ricorsiva
Il modello implementa un **forecasting ricorsivo multi-step**:

```python
def recursive_forecast_xgb(model, x_start, n_forecast):
    xinput = x_start.copy()
    forecast = []
    
    for step in range(n_forecast):
        pred = model.predict(xinput.reshape(1, -1))[0]
        forecast.append(pred)
        xinput = np.roll(xinput, -1)  # Shift della finestra
        xinput[-1] = pred             # Inserimento predizione
```

### 5.2 Processo Step-by-Step
1. **Inizializzazione**: Ultime 104 settimane del training set
2. **Predizione**: Genera forecast per t+1
3. **Update Window**: Rimuove valore più vecchio, aggiunge predizione
4. **Iterazione**: Ripete per tutte le 52 settimane del 2024

### 5.3 Vantaggi e Limitazioni

#### Vantaggi:
- Utilizza informazioni aggiornate dalle proprie predizioni
- Flessibile per orizzonti temporali variabili
- Mantiene la struttura temporale della finestra

#### Limitazioni:
- Propagazione degli errori nelle predizioni future
- Perdita di informazioni storiche oltre la finestra
- Instabilità crescente con l'orizzonte temporale

## 6. Risultati del Modello

### 6.1 Performance di Training
```
Dataset creato: X shape = (416, 104), y shape = (416,)
Usando 104 settimane per predire la settimana successiva
Modello addestrato con successo!
```

### 6.2 Metriche di Accuratezza

#### Risultati Ottenuti:
```
--- RISULTATI ACCURATEZZA XGBOOST ---
MAPE (Mean Absolute Percentage Error): 0.3154 (31.54%)
ME (Mean Error): -0.9583
MAE (Mean Absolute Error): 4.6893
MPE (Mean Percentage Error): -0.0049 (-0.49%)
RMSE (Root Mean Square Error): 6.1352
Correlazione: 0.7328
```

#### Interpretazione delle Metriche:

**Metriche di Errore:**
- **MAE = 4.69 mm**: Errore medio assoluto accettabile
- **RMSE = 6.14 mm**: Errore quadratico medio buono
- **ME = -0.96 mm**: Bias negativo molto piccolo (leggera sottostima)
- **MAPE = 31.54%**: Errore percentuale moderato

**Metriche di Correlazione:**
- **Correlazione = 0.7328**: Buona correlazione tra predetto e osservato
- **MPE = -0.49%**: Bias percentuale quasi nullo

### 6.3 Analisi Comparativa delle Performance

#### Punti di Forza:
- **Bias minimo**: ME e MPE molto bassi
- **Stabilità**: Errori consistenti senza derive sistematiche
- **Robustezza**: Buone performance generali su diverse metriche
- **Efficienza**: Training veloce e predizioni rapide

#### Aree di Miglioramento:
- **MAPE**: Ancora relativamente medio (31.54%)
- **Correlazione**: Buona ma non eccellente (0.73)
- **Variabilità**: RMSE indica presenza di errori occasionali significativi

## 7. Visualizzazioni Generate
![graficoSarima.png](img/graficoXGBoost.png)
### 7.1 Grafici di Training
- **Training Loss**: Andamento della Huber Loss durante le 1000 epoche
- **Convergenza**: Focus sulle ultime 100 epoche per analizzare la stabilizzazione

### 7.2 Grafici di Forecasting
- **In-Sample Predictions**: Sovrapposizione tra dati reali e predizioni sui dati di training
- **Out-of-Sample Forecast**: Confronto tra dati reali 2024 e previsioni della rete neurale




