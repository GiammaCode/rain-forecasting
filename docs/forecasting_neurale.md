# Documentazione Modello Rete Neurale - Deep Learning per Forecasting Piogge

## Panoramica

Il modulo `NN_model.py` implementa una Rete Neurale Feedforward (Multi-Layer Perceptron) per il forecasting delle piogge settimanali dell'Emilia-Romagna utilizzando PyTorch. Questo documento descrive l'architettura del modello, i parametri utilizzati, i risultati ottenuti e l'interpretazione delle performance.

## 1. Introduzione al Modello di Rete Neurale

### 1.1 Definizione
Una Rete Neurale Feedforward è un modello di machine learning che simula il funzionamento dei neuroni biologici attraverso layers interconnessi di nodi artificiali. È particolarmente efficace per catturare relazioni non lineari complesse nei dati temporali.

### 1.2 Architettura del Modello
Il modello implementa una architettura Multi-Layer Perceptron (MLP) con:
- **Input Layer**: 104 neuroni (corrispondenti a 104 settimane di lookback)
- **Hidden Layers**: Due layers nascosti con attivazione ReLU
- **Output Layer**: 1 neurone per la predizione della settimana successiva

## 2. Architettura Dettagliata

### 2.1 Struttura del Modello
```python
RainForecastNN(
  (network): Sequential(
    (0): Linear(in_features=104, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=32, bias=True)
    (3): ReLU()
    (4): Linear(in_features=32, out_features=1, bias=True)
  )
)
```

### 2.2 Componenti dell'Architettura

#### Layer 1 - Input Layer:
- **Input Size**: 104 (corrispondente a 2 anni di dati settimanali)
- **Output Size**: 64 neuroni
- **Attivazione**: ReLU (Rectified Linear Unit)

#### Layer 2 - Hidden Layer:
- **Input Size**: 64 neuroni
- **Output Size**: 32 neuroni
- **Attivazione**: ReLU

#### Layer 3 - Output Layer:
- **Input Size**: 32 neuroni
- **Output Size**: 1 neurone (predizione finale)
- **Attivazione**: Lineare (nessuna attivazione)

### 2.3 Funzioni di Attivazione

**ReLU (Rectified Linear Unit)**:
```
f(x) = max(0, x)
```
- Vantaggi: Computazionalmente efficiente, evita il problema del vanishing gradient
- Applicata nei layers nascosti per introdurre non linearità

## 3. Parametri di Training

### 3.1 Configurazione di Training

```python
# Iperparametri principali
learning_rate = 0.0007
n_epochs = 1000
batch_size = 8
look_back = 104  # 2 anni di dati
```

### 3.2 Funzione di Loss e Ottimizzatore

#### Loss Function - Huber Loss:
```python
loss_fn = nn.HuberLoss(delta=3.0)
```
- **Caratteristiche**: Combina vantaggi di MSE e MAE
- **Delta**: 3.0 (soglia per passare da quadratica a lineare)
- **Vantaggi**: Robusta agli outliers, convergenza stabile

#### Ottimizzatore - Adam:
```python
optimizer = optim.Adam(model.parameters(), lr=0.0007)
```
- **Tipo**: Adaptive Moment Estimation
- **Learning Rate**: 0.0007
- **Vantaggi**: Adattamento automatico del learning rate

### 3.3 Preparazione Dati

#### Dataset Creation:
- **Lookback Window**: 104 settimane (2 anni)
- **Dataset Shape**: X (416, 104), y (416,)
- **Approccio**: Supervised learning con finestra scorrevole

## 4. Processo di Training

### 4.1 Training Loop
1. **Mini-batch processing** con batch size di 8
2. **Forward pass** attraverso la rete
3. **Calcolo loss** con Huber Loss
4. **Backward propagation** per calcolare gradienti
5. **Update parametri** con ottimizzatore Adam

### 4.2 Convergenza del Training

#### Andamento della Loss:
```
Epoch 0, Loss: 31.128900
Epoch 100, Loss: 5.072647
Epoch 200, Loss: 1.351589
Epoch 300, Loss: 1.050124
Epoch 400, Loss: 0.604659
Epoch 500, Loss: 1.864790
Epoch 600, Loss: 1.585737
Epoch 700, Loss: 0.552764
Epoch 800, Loss: 0.482703
Epoch 900, Loss: 0.495571
Epoch 999, Loss: 0.512198
```

#### Interpretazione:
- **Rapida convergenza iniziale**: Da 31.13 a 5.07 in 100 epoche
- **Stabilizzazione**: Loss si stabilizza intorno a 0.5-1.0
- **Oscillazioni**: Normali dovute al batch size piccolo e alla natura stocastica

## 5. Forecasting Ricorsivo

### 5.1 Metodologia
Il modello utilizza un approccio **ricorsivo multi-step**:

1. **Inizializzazione**: Ultimi 104 valori del training set
2. **Predizione**: Genera previsione per t+1
3. **Update**: Sostituisce il valore più vecchio con la predizione
4. **Iterazione**: Ripete per tutte le 52 settimane del 2024

### 5.2 Vantaggi e Svantaggi

#### Vantaggi:
- Utilizza le proprie predizioni per step futuri
- Adattabile a orizzonti temporali variabili

#### Svantaggi:
- Accumulo di errori nelle predizioni a lungo termine
- Dipendenza dalla qualità delle prime predizioni

## 6. Risultati del Modello

### 6.1 Metriche di Accuratezza

#### Risultati Ottenuti:
```
--- RISULTATI ACCURATEZZA RETE NEURALE ---
MAPE (Mean Absolute Percentage Error): 0.3320 (33.20%)
ME (Mean Error): -4.2478
MAE (Mean Absolute Error): 5.4163
MPE (Mean Percentage Error): -0.2063 (-20.63%)
RMSE (Root Mean Square Error): 6.9170
Correlazione: 0.7712
```

#### Interpretazione delle Metriche:

**Metriche di Errore:**
- **MAE = 5.42 mm**: Errore medio assoluto moderato
- **RMSE = 6.92 mm**: Errore quadratico medio accettabile
- **ME = -4.25 mm**: Bias negativo significativo (sottostima sistematica)
- **MAPE = 33.20%**: Errore percentuale medio

**Metriche di Correlazione:**
- **Correlazione = 0.7712**: Buona correlazione tra predetto e osservato
- **MPE = -20.63%**: Tendenza sistematica alla sottostima

### 6.2 Analisi delle Performance

#### Punti di Forza:
- Buona capacità di catturare trend generali (correlazione 0.77)
- Architettura semplice ma efficace
- Convergenza stabile durante il training

#### Punti di Debolezza:
- Bias sistematico verso la sottostima
- MAPE discreto (33.20%)
- Accumulo di errori nel forecasting ricorsivo

## 7. Visualizzazioni Generate
![graficoSarima.png](img/graficoNN.png)
### 7.1 Grafici di Training
- **Training Loss**: Andamento della Huber Loss durante le 1000 epoche
- **Convergenza**: Focus sulle ultime 100 epoche per analizzare la stabilizzazione

### 7.2 Grafici di Forecasting
- **In-Sample Predictions**: Sovrapposizione tra dati reali e predizioni sui dati di training
- **Out-of-Sample Forecast**: Confronto tra dati reali 2024 e previsioni della rete neurale

## 8. Considerazioni Tecniche

### 8.1 Riproducibilità
Il modello implementa controlli per la riproducibilità:
```python
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
```

### 8.2 Gestione del Overfitting
- **Architecture**: Relativamente semplice per evitare overfitting
- **Early Stopping**: Implementato ma disabilitato nel codice finale
- **Batch Size**: Piccolo (8) per regolarizzazione implicita
