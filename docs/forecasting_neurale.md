---
layout: default
title: RNN
nav_order: 4
---
# Previsione di Pioggia Settimanale - Rete Neurale (2024)

## Obiettivo del Progetto

Questo progetto mira a prevedere la quantità di pioggia settimanale in millimetri per l'anno 2024 mediante una **rete neurale artificiale feedforward**. L'approccio si basa su una finestra mobile (look-back) di 52 settimane per apprendere i pattern storici delle precipitazioni.

---

## Teoria del Modello Neurale

La rete neurale implementata è una **rete densa (fully connected)** con tre strati nascosti. Utilizza la tecnica del **forecasting ricorsivo**, in cui ciascuna previsione è utilizzata come input per predire il valore successivo.

### Architettura del modello:

```
NeuralForecaster(
    input_size=52
    -> Linear(52 -> 32) -> ReLU
    -> Linear(32 -> 16) -> ReLU
    -> Linear(16 -> 1)
)
```

**Addestramento**:
- Ottimizzatore: Adam 
- Funzione di perdita: MSELoss 
- Epoche: 1000 
- Learning rate: 0.0007 
- Batch size: 1

**Forecasting ricorsivo:**
A partire dalle ultime 52 settimane del set di addestramento, il modello genera previsioni passo-passo per l’intero 
anno successivo (52 settimane).

### Visualizzazione dei Risultati
![NNplot.png](img/NNplot.png)

- Linea blu: pioggia reale del 2024 (test)

- Linea arancione: previsione effettuata dalla rete neurale

### Metriche di Accuratezza

| Metrica   | Valore |
| --------- |--------|
| MAPE (%)  | 1.90   |
| MAE (mm)  | 8.98   |
| RMSE (mm) | 11.92  |
| Corr      | 0.30   |
| ME        | -0.89  |
| MPE (%)   | 1.46   |


**Interpretazione**:
- Il modello prevede molto bene in media: valori vicini a quelli reali (MAPE e ME molto buoni).

- È ancora timido nei picchi: alcuni salti non vengono seguiti.

- La correlazion di 0.30 è  segno che la rete inizia a capire l'andamento, 
ma non lo segue ancora con decisione.

La rete neurale sviluppata ha mostrato una buona capacità di previsione in termini di errore assoluto, 
ma una correlazione temporale più debole rispetto al modello SARIMAX.
